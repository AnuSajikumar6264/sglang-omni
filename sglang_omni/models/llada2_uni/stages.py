# SPDX-License-Identifier: Apache-2.0
"""Stage factories for LLaDA2-Uni pipeline."""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.models.llada2_uni.config import IMAGE_STAGE, THINKER_STAGE

logger = logging.getLogger(__name__)


def _event_to_dict(event) -> dict[str, Any]:
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": bool(event.is_final),
    }


def create_preprocessing_executor(
    model_path: str,
    *,
    max_seq_len: int | None = None,
):
    from sglang_omni.models.llada2_uni.components.preprocessor import LLaDA2Preprocessor
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    preprocessor = LLaDA2Preprocessor(
        model_path=model_path,
        max_seq_len=max_seq_len,
    )
    return SimpleScheduler(preprocessor)


def create_image_encoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: Any = None,
):
    import torch

    from sglang_omni.models.llada2_uni.components.image_encoder import (
        LLaDA2ImageEncoder,
    )
    from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
    from sglang_omni.models.llada2_uni.request_builders import (
        apply_encoder_result,
        build_encoder_request,
        merge_image_tokens_for_thinker,
    )
    from sglang_omni.models.weight_loader import resolve_dtype
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
    from sglang_omni.utils.device import resolve_concrete_device

    dtype = resolve_dtype(dtype)
    device = str(resolve_concrete_device(device, gpu_id))

    model = LLaDA2ImageEncoder(model_path=model_path, device=device, dtype=dtype)

    def _encode(payload):
        state = LLaDA2UniPipelineState.from_dict(payload.data)
        request = build_encoder_request(state, stage_name=IMAGE_STAGE)

        if request.get("_skip"):
            result = request.get("_result", {})
        else:
            with torch.no_grad():
                result = model(**request)

        apply_encoder_result(state, stage_name=IMAGE_STAGE, result=result)
        merge_image_tokens_for_thinker(state)
        state.encoder_inputs.clear()
        state.encoder_outs.clear()
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(_encode)


def create_sglang_dllm_thinker_executor_from_config(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    max_seq_len: int = 8192,
    dllm_algorithm: str = "LowConfidence",
    dllm_algorithm_config: str | None = None,
    server_args_overrides: dict[str, Any] | None = None,
):
    """Create an DllmScheduler for the LLaDA2-Uni thinker."""
    from sglang_omni.models.llada2_uni.bootstrap import create_dllm_thinker_scheduler
    from sglang_omni.platforms import current_platform
    from sglang_omni.scheduling.sglang_backend import (
        build_sglang_server_args,
        pin_resolved_device_type,
    )
    from sglang_omni.utils.device import resolve_concrete_device

    concrete_device = resolve_concrete_device(device, gpu_id)
    resolved_gpu_id = concrete_device.index or 0

    stated = dict(server_args_overrides or {})
    platform_backend = current_platform.get_dllm_attention_backend()
    overrides: dict[str, Any] = {
        "attention_backend": platform_backend or "flashinfer",
        "sampling_backend": "pytorch",
        "tp_size": tp_size,
    }
    if not current_platform.enable_dllm_decode_graph():
        overrides["disable_cuda_graph"] = True
    overrides.update(stated)
    pin_resolved_device_type(overrides, concrete_device.type)

    if platform_backend and not overrides.get("disable_cuda_graph", False):
        # SGLang's dLLM resolution renames the backend to flashinfer for any dLLM
        # that captures decode graphs, and branches only for ROCm and NPU, so a
        # platform outside that set would lose the backend its blocks need on the
        # way in. Refuse at configuration time rather than at the first forward.
        raise ValueError(
            f"{current_platform.device_type} requires attention_backend="
            f"{platform_backend!r} for a diffusion LLM, but SGLang's dLLM "
            "resolution replaces it with flashinfer whenever decode graphs are "
            "captured; keep enable_dllm_decode_graph() False until that pass "
            "knows this platform"
        )

    server_args = build_sglang_server_args(
        model_path,
        context_length=max_seq_len,
        dllm_algorithm=dllm_algorithm,
        dllm_algorithm_config=dllm_algorithm_config,
        **overrides,
    )
    from sglang.srt.arg_groups.model_override_base import resolved_view

    cfg = resolved_view(server_args)
    # The graph state is the request this factory made, not cfg.cuda_graph_config:
    # SGLang folds the graph switches into that field in its resolution pass, which
    # runs inside the engine bootstrap below, and a view read before that answers
    # with the raw input -- still None.
    logger.info(
        "create_sglang_dllm_thinker_executor_from_config: "
        "dllm_algorithm=%s, tp_rank=%s/%s, attention_backend=%s, "
        "decode_cuda_graph=%s, mem_fraction_static=%s",
        cfg.dllm_algorithm,
        tp_rank,
        tp_size,
        cfg.attention_backend,
        not overrides.get("disable_cuda_graph", False),
        cfg.mem_fraction_static,
    )
    return create_dllm_thinker_scheduler(
        server_args,
        resolved_gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
    )


def create_decode_executor(model_path: str):
    from sglang_omni.models.llada2_uni.components.common import load_llada2_tokenizer
    from sglang_omni.models.llada2_uni.merge import decode_events
    from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    tokenizer = load_llada2_tokenizer(model_path)

    def _decode(payload):
        state = LLaDA2UniPipelineState.from_dict(payload.data)
        thinker_out = state.thinker_out or state.engine_outputs.get(THINKER_STAGE)
        if not isinstance(thinker_out, dict):
            logger.warning(
                "request %s: thinker produced no output (got %s), returning empty text",
                payload.request_id,
                type(thinker_out).__name__,
            )
            thinker_out = {
                "output_ids": [],
                "is_final": True,
            }

        events = decode_events(
            thinker_out=thinker_out,
            tokenizer=tokenizer,
        )
        event_dicts = [_event_to_dict(event) for event in events]

        result: dict[str, Any] = {"events": event_dicts}
        if events:
            result.update(events[0].payload)
            result.setdefault("modality", events[0].modality)

        finish_reason = thinker_out.get("finish_reason")
        if finish_reason is not None:
            result.setdefault("finish_reason", finish_reason)

        input_ids = (
            state.prompt.get("input_ids") if isinstance(state.prompt, dict) else None
        )
        if input_ids is None:
            prompt_tokens = 0
        elif hasattr(input_ids, "numel"):
            prompt_tokens = int(input_ids.numel())
        else:
            prompt_tokens = len(input_ids)

        completion_ids = thinker_out.get("output_ids") or []
        completion_tokens = len(completion_ids)

        result.setdefault(
            "usage",
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

        payload.data = result
        return payload

    return SimpleScheduler(_decode)
