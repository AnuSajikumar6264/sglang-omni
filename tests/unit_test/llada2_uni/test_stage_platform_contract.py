# SPDX-License-Identifier: Apache-2.0
"""What the LLaDA2-Uni stages must ask of the platform they land on.

The pipeline was written against CUDA: the image encoder took a literal
``device="cuda"`` and the thinker named flashinfer unconditionally. Both are
invisible on a CUDA host and fatal anywhere else, and neither is reachable from
the model tests, which need real checkpoints. So the factories are driven here
with the heavy dependencies patched out and their platform-facing decisions
observed directly.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sglang.srt.model_executor.cuda_graph_config import Backend

import sglang_omni.platforms as platforms
from sglang_omni.models.llada2_uni import stages
from sglang_omni.models.llada2_uni.config import IMAGE_STAGE, THINKER_STAGE, EntryClass
from tests.unit_test.fixtures.mini_checkpoint import write_mini_llama_checkpoint


class _FakePlatform:
    """A platform stand-in carrying only the hooks these stages consult."""

    def __init__(
        self,
        *,
        dllm_backend: str | None,
        dllm_graph: bool,
        device_type: str = "xpu",
        decode_graph_backend: str | None = None,
    ) -> None:
        self.device_type = device_type
        self._dllm_backend = dllm_backend
        self._dllm_graph = dllm_graph
        self._decode_graph_backend = decode_graph_backend

    def get_dllm_attention_backend(self) -> str | None:
        return self._dllm_backend

    def enable_dllm_decode_graph(self) -> bool:
        return self._dllm_graph

    def get_decode_cuda_graph_backend(self) -> str | None:
        return self._decode_graph_backend


def _drive_thinker(
    monkeypatch: pytest.MonkeyPatch,
    platform: _FakePlatform,
    **factory_kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the thinker against a fake platform; return what it asked for."""
    from sglang_omni.models.llada2_uni import bootstrap
    from sglang_omni.scheduling import sglang_backend

    build_kwargs: dict[str, Any] = {}
    scheduler_kwargs: dict[str, Any] = {}

    def fake_build(model_path, **kwargs):
        del model_path
        build_kwargs.update(kwargs)
        # The real builder resolves before it returns, so the stand-in answers
        # with a resolved cuda_graph_config -- and mirrors the precedence the
        # stage depends on: a per-phase decode backend outranks the legacy
        # disable switch. The end-to-end tests below pin that against SGLang.
        decode = kwargs.get("cuda_graph_backend_decode")
        if decode is None:
            decode = (
                Backend.DISABLED if kwargs.get("disable_cuda_graph") else Backend.FULL
            )
        return SimpleNamespace(
            attention_backend=kwargs.get("attention_backend"),
            dllm_algorithm=kwargs.get("dllm_algorithm"),
            mem_fraction_static=None,
            disable_cuda_graph=bool(kwargs.get("disable_cuda_graph", False)),
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend=decode),
                prefill=SimpleNamespace(backend=Backend.DISABLED),
            ),
        )

    def fake_scheduler(server_args, gpu_id, **kwargs):
        scheduler_kwargs.update(
            {"server_args": server_args, "gpu_id": gpu_id, **kwargs}
        )
        return SimpleNamespace()

    monkeypatch.setattr(platforms, "current_platform", platform)
    monkeypatch.setattr(sglang_backend, "build_sglang_server_args", fake_build)
    monkeypatch.setattr(bootstrap, "create_dllm_thinker_scheduler", fake_scheduler)

    stages.create_sglang_dllm_thinker_executor_from_config("unused", **factory_kwargs)
    return build_kwargs, scheduler_kwargs


def _write_mini_dllm_checkpoint(tmp_path: Path) -> str:
    """A checkpoint the real resolution pipeline accepts as a diffusion LLM."""
    return write_mini_llama_checkpoint(tmp_path, architectures=["LLaDA2MoeModelLM"])


def _drive_real_thinker(
    monkeypatch: pytest.MonkeyPatch,
    platform: _FakePlatform,
    model_path: str,
    **overrides: Any,
) -> dict[str, Any]:
    """Build the thinker through SGLang's own resolution; fake only the engine.

    The device is pinned to cuda the way upstream's resolution tests pin it, so
    the record resolves the same on an accelerator-less host, and the platform
    stand-in claims cuda for the same reason -- what is under test is a platform
    that names a dLLM backend, not XPU's device support.

    gpu_id is passed explicitly because placement always passes it for a GPU
    stage: without an index resolve_concrete_device asks the host which card
    this process is on, which raises on a host whose torch has no cuda.
    """
    from sglang_omni.models.llada2_uni import bootstrap

    scheduler_kwargs: dict[str, Any] = {}

    def fake_scheduler(server_args, gpu_id, **kwargs):
        scheduler_kwargs.update(
            {"server_args": server_args, "gpu_id": gpu_id, **kwargs}
        )
        return SimpleNamespace()

    monkeypatch.setattr(platforms, "current_platform", platform)
    monkeypatch.setattr(bootstrap, "create_dllm_thinker_scheduler", fake_scheduler)

    stages.create_sglang_dllm_thinker_executor_from_config(
        model_path,
        device="cuda",
        gpu_id=0,
        max_seq_len=2048,
        server_args_overrides=dict(overrides),
    )
    return scheduler_kwargs


def test_no_stage_pins_a_device_in_the_pipeline_config() -> None:
    """The config must not name a device; placement is the host's to decide."""
    config = EntryClass(model_path="unused")

    assert config.stage_named(IMAGE_STAGE).factory.device is None
    assert config.stage_named(THINKER_STAGE).factory.device is None


def test_the_image_encoder_resolves_an_absent_device_to_its_placed_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """device=None must reach the platform, carrying the stage's gpu id.

    An unresolved None reaches torch.device(), which raises; a literal "cuda"
    would raise on any other accelerator instead of running there.
    """
    from sglang_omni.models.llada2_uni.components import image_encoder

    built: dict[str, Any] = {}

    class _Encoder:
        def __init__(self, *, model_path, device, dtype):
            del model_path, dtype
            built["device"] = device

    monkeypatch.setattr(image_encoder, "LLaDA2ImageEncoder", _Encoder)

    stages.create_image_encoder_executor("unused", device=None, gpu_id=1)

    live = platforms.current_platform.device_type
    assert built["device"] == ("cpu" if live == "cpu" else f"{live}:1")


def test_the_thinker_asks_for_the_backend_its_platform_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dLLM block is bidirectional, so the backend is not a free choice.

    A platform SGLang's own dLLM resolution does not cover names its own; the
    rest keep the flashinfer request this stage has always made.
    """
    named, _ = _drive_thinker(
        monkeypatch, _FakePlatform(dllm_backend="triton", dllm_graph=False)
    )
    assert named["attention_backend"] == "triton"

    unnamed, _ = _drive_thinker(
        monkeypatch, _FakePlatform(dllm_backend=None, dllm_graph=False)
    )
    assert unnamed["attention_backend"] == "flashinfer"


def test_a_platform_named_backend_refuses_the_capture_that_would_replace_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SGLang's dLLM pass renames the backend to flashinfer for any dLLM that
    captures decode graphs, and branches only for ROCm and NPU. A platform
    outside that set would silently lose the backend its bidirectional blocks
    need, so the combination is refused while it is still configuration."""
    with pytest.raises(ValueError, match="flashinfer"):
        _drive_thinker(
            monkeypatch, _FakePlatform(dllm_backend="triton", dllm_graph=True)
        )


def test_the_platforms_sglang_covers_keep_its_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On ROCm and NPU that rewrite is the correct one (triton, ascend), so a
    stage that named nothing must not stand in its way."""
    named, _ = _drive_thinker(
        monkeypatch, _FakePlatform(dllm_backend=None, dllm_graph=True)
    )

    assert named["attention_backend"] == "flashinfer"
    assert "disable_cuda_graph" not in named


def test_an_operator_named_backend_does_not_escape_the_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rewrite replaces whatever backend is on the way in, so naming one by
    hand does not survive capture either -- disabling capture alongside it does,
    and then the operator's choice reaches SGLang untouched."""
    with pytest.raises(ValueError, match="flashinfer"):
        _drive_thinker(
            monkeypatch,
            _FakePlatform(dllm_backend="triton", dllm_graph=True),
            server_args_overrides={"attention_backend": "intel_xpu"},
        )

    named, _ = _drive_thinker(
        monkeypatch,
        _FakePlatform(dllm_backend="triton", dllm_graph=True),
        server_args_overrides={
            "attention_backend": "intel_xpu",
            "disable_cuda_graph": True,
        },
    )
    assert named["attention_backend"] == "intel_xpu"


def test_graph_capture_follows_the_platform_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dLLM stage has run eager everywhere since it landed; a platform opts
    in once it has validated capture, and only for itself."""
    eager, _ = _drive_thinker(
        monkeypatch, _FakePlatform(dllm_backend=None, dllm_graph=False)
    )
    assert eager["disable_cuda_graph"] is True

    captured, _ = _drive_thinker(
        monkeypatch, _FakePlatform(dllm_backend=None, dllm_graph=True)
    )
    assert "disable_cuda_graph" not in captured


def test_a_stated_graph_switch_outranks_the_platform_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """engine.disable_cuda_graph is how an operator debugs a capture problem, so
    it has to win over the platform's default on either side."""
    forced_off, _ = _drive_thinker(
        monkeypatch,
        _FakePlatform(dllm_backend="triton", dllm_graph=True),
        server_args_overrides={"disable_cuda_graph": True},
    )
    assert forced_off["disable_cuda_graph"] is True
    assert forced_off["attention_backend"] == "triton"

    forced_on, _ = _drive_thinker(
        monkeypatch,
        _FakePlatform(dllm_backend=None, dllm_graph=False),
        server_args_overrides={"disable_cuda_graph": False},
    )
    assert forced_on["disable_cuda_graph"] is False


def test_the_thinker_passes_its_tp_identity_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 16B thinker does not fit one 24 GiB card, so the stage runs under TP.

    A TP stage's factory is called with tp_rank / tp_size / nccl_port injected;
    tp_size has to reach SGLang and the rank identity the scheduler.
    """
    named, scheduler = _drive_thinker(
        monkeypatch,
        _FakePlatform(dllm_backend="triton", dllm_graph=False),
        gpu_id=2,
        tp_rank=1,
        tp_size=2,
        nccl_port=29500,
    )

    assert named["tp_size"] == 2
    assert scheduler["gpu_id"] == 2
    assert scheduler["tp_rank"] == 1
    assert scheduler["nccl_port"] == 29500


def test_the_resolved_decode_backend_is_never_the_field_on_server_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``server_args.cuda_graph_config`` stays None through a whole build.

    SGLang's resolver for it is declaration-only: it never writes the field, so
    the graph switches are readable only through the accessor that overlays the
    declarations. A field read answers None on a fully resolved record and kills
    the stage at construction -- ``AttributeError: 'NoneType' object has no
    attribute 'decode'`` -- before any weight is touched, on every platform.
    """
    from sglang_omni.scheduling.generation_batch_policy import (
        get_decode_cuda_graph_backend,
    )

    scheduler = _drive_real_thinker(
        monkeypatch,
        _FakePlatform(dllm_backend="triton", dllm_graph=False, device_type="cuda"),
        _write_mini_dllm_checkpoint(tmp_path),
    )
    server_args = scheduler["server_args"]

    assert server_args._resolution_finished is True
    assert server_args.cuda_graph_config is None
    assert get_decode_cuda_graph_backend(server_args) == Backend.DISABLED


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"cuda_graph_backend_decode": "full"}, id="per-phase-backend"),
        pytest.param(
            {"cuda_graph_config": {"decode": {"backend": "full"}}}, id="nested-config"
        ),
    ],
)
def test_capture_turned_on_past_the_disable_switch_still_meets_the_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, overrides: dict[str, Any]
) -> None:
    """Both of SGLang's newer graph switches outrank ``disable_cuda_graph``.

    The stage requests the disable switch to stay eager, but a per-phase backend
    and an explicit cuda_graph_config are both resolved after it and win, so a
    refusal that read the request would pass the run straight through to the
    flashinfer rewrite it exists to prevent.
    """
    with pytest.raises(ValueError, match="flashinfer"):
        _drive_real_thinker(
            monkeypatch,
            _FakePlatform(dllm_backend="triton", dllm_graph=False, device_type="cuda"),
            _write_mini_dllm_checkpoint(tmp_path),
            **overrides,
        )


def test_a_platform_that_defaults_decode_capture_on_meets_the_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A platform names its decode graph backend for every SGLang stage it hosts.

    That default is applied to this stage too, so dropping the dLLM stage's own
    disable switch would hand the run capture -- and the flashinfer rewrite --
    from a hook that says nothing about diffusion LLMs.
    """
    platform = _FakePlatform(
        dllm_backend="triton",
        dllm_graph=True,
        device_type="cuda",
        decode_graph_backend=Backend.FULL,
    )
    with pytest.raises(ValueError, match="flashinfer"):
        _drive_real_thinker(
            monkeypatch, platform, _write_mini_dllm_checkpoint(tmp_path)
        )


def test_the_thinker_takes_the_memory_fraction_its_stage_was_placed_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The placement fraction reaches a factory only if it declares the kwarg.

    resolve_factory_signature_args injects it where the signature asks and drops
    it in silence otherwise, so an undeclared thinker let SGLang size its KV pool
    against the whole card -- while the image encoder held the rest of it.
    """
    from sglang_omni.config.runtime import resolve_stage_factory_args

    config = EntryClass(model_path="unused")
    stage_cfg = config.stage_named(THINKER_STAGE)
    stage_cfg.gpu_memory_fraction = 0.8

    resolved = resolve_stage_factory_args(stage_cfg, config, gpu_id=0)
    assert resolved["total_gpu_memory_fraction"] == 0.8

    _, scheduler = _drive_thinker(
        monkeypatch,
        _FakePlatform(dllm_backend="triton", dllm_graph=False),
        total_gpu_memory_fraction=0.8,
    )
    assert scheduler["total_gpu_memory_fraction"] == 0.8


def test_the_thinker_refuses_a_tp_size_the_pipeline_did_not_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tp_size is placement, not a request: the followers are already spawned.

    An override that disagreed used to be dropped silently -- the factory seeded
    tp_size first and let the operator's value overwrite it -- so SGLang built a
    group of a different width than the stage had ranks for and the first MoE
    collective hung.
    """
    with pytest.raises(ValueError, match="tp_size"):
        _drive_thinker(
            monkeypatch,
            _FakePlatform(dllm_backend="triton", dllm_graph=False),
            tp_size=2,
            server_args_overrides={"tp_size": 4},
        )

    named, _ = _drive_thinker(
        monkeypatch,
        _FakePlatform(dllm_backend="triton", dllm_graph=False),
        tp_size=2,
        server_args_overrides={"tp_size": 2},
    )
    assert named["tp_size"] == 2


def test_the_dllm_scheduler_declares_that_tp_needs_work_replication() -> None:
    """It holds no TP group to broadcast over, unlike OmniScheduler, so without
    replication a follower rank would idle while its peers entered the MoE
    collectives. The stage reads the flag with a False default, so staying
    silent here means silently running rank 0 only."""
    from sglang_omni.scheduling.dllm_scheduler import DllmScheduler

    scheduler = DllmScheduler.__new__(DllmScheduler)
    DllmScheduler.__init__(
        scheduler,
        tp_worker=SimpleNamespace(),
        tree_cache=SimpleNamespace(),
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=SimpleNamespace(),
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(),
        dllm_config=SimpleNamespace(block_size=32),
        request_builder=lambda data: data,
        result_adapter=lambda data: data,
    )

    assert scheduler.requires_tp_work_fanout is True


def test_every_platform_answers_the_hooks_this_model_reads() -> None:
    """The stage consults two hooks on whatever platform it lands on. A platform
    that answered neither would fall back to the base class, which leaves the
    backend to SGLang's own dLLM pass; XPU is the one that names it, because that
    pass has no XPU branch and would hand the run CUDA-only flashinfer.

    Decode capture stays off everywhere: no attention backend reachable on XPU
    accepts ForwardMode.DLLM_EXTEND in a graph, and on CUDA this stage has not
    been measured with capture on.
    """
    from sglang_omni.platforms.cuda import CUDAOmniPlatform
    from sglang_omni.platforms.interface import OmniPlatform
    from sglang_omni.platforms.rocm import ROCMOmniPlatform
    from sglang_omni.platforms.xpu import XPUOmniPlatform

    expected = {
        OmniPlatform: (None, False),
        CUDAOmniPlatform: (None, False),
        ROCMOmniPlatform: (None, False),
        XPUOmniPlatform: ("triton", False),
    }
    for platform_class, (backend, graph) in expected.items():
        platform = platform_class()
        assert platform.get_dllm_attention_backend() == backend, platform_class.__name__
        assert platform.enable_dllm_decode_graph() is graph, platform_class.__name__
