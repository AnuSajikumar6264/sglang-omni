# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for AURA Omni: Qwen3-ASR, AURA (Qwen3-VL) and Qwen3-TTS."""

from __future__ import annotations

from typing import ClassVar, TypedDict

from pydantic import Field

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.platforms import current_platform

PACKAGE = "sglang_omni.models.aura_omni_av"
ASR_PACKAGE = "sglang_omni.models.qwen3_asr"
TTS_PACKAGE = "sglang_omni.models.qwen3_tts"
ROUTING = f"{PACKAGE}.request_builders"

TTS_STAGES = ("tts_preprocessing", "tts_engine", "vocoder")
# Qwen3-ASR and Qwen3-TTS share card 0 from two processes, so each engine gets a fixed KV pool that does not depend on which one starts first.
SPEECH_KV_CACHE_BYTES = 1 << 30


class AuraOmniStageFactoryKwargs(TypedDict, total=False):
    model_path: str
    async_decode: bool


def aura_omni_av_stages() -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="preprocessing",
            factory_path=f"{PACKAGE}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_concurrency=8),
            next=["asr", "aura"],
            route_fn=f"{ROUTING}.resolve_preprocessing_next_stages",
            project_payload={
                "asr": f"{ROUTING}.project_preprocessing_to_asr",
                "aura": f"{ROUTING}.project_preprocessing_to_aura",
            },
        ),
        EngineStageConfig(
            name="asr",
            process="asr",
            factory_path=f"{ASR_PACKAGE}.stages.create_sglang_qwen3_asr_executor",
            engine=EngineArgs(
                max_running_requests=8,
                enable_torch_compile=False,
                kv_cache_bytes=SPEECH_KV_CACHE_BYTES,
            ),
            gpu=0,
            # Measured peak 7.9 GiB of a 23.9 GiB Arc Pro B60.
            gpu_memory_fraction=0.35,
            next="aura",
        ),
        EngineStageConfig(
            name="aura",
            process="aura",
            factory_path=f"{PACKAGE}.stages.create_aura_executor",
            factory=FactoryArgs(max_new_tokens=256, request_build_max_workers=2),
            gpu=1,
            wait_for=["preprocessing", "asr"],
            wait_for_fn=f"{ROUTING}.resolve_aura_wait_sources",
            merge_fn=f"{ROUTING}.merge_for_aura",
            next=["decode", "tts_preprocessing", "vocoder"],
            route_fn=f"{ROUTING}.resolve_aura_next_stages",
            stream_to=["decode", "vocoder"],
            stream_done_to_fn=f"{ROUTING}.resolve_aura_stream_done_targets",
            project_payload={
                "tts_preprocessing": f"{ROUTING}.project_aura_to_tts",
                "vocoder": f"{ROUTING}.project_aura_to_vocoder",
            },
        ),
        StageConfig(
            name="decode",
            process="aura",
            factory_path=f"{PACKAGE}.stages.create_decode_executor",
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
        StageConfig(
            name="tts_preprocessing",
            process="tts",
            factory_path=f"{TTS_PACKAGE}.stages.create_preprocessing_executor",
            next="tts_engine",
        ),
        # The vocoder is built before the engine it shares a process with, so its weights are resident when the KV pool is sized.
        StageConfig(
            name="vocoder",
            process="tts",
            factory_path=f"{TTS_PACKAGE}.stages.create_vocoder_executor",
            factory=FactoryArgs(dtype="bfloat16"),
            gpu=0,
            gpu_memory_fraction=0.15,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
        EngineStageConfig(
            name="tts_engine",
            process="tts",
            factory_path=f"{TTS_PACKAGE}.stages.create_sglang_tts_engine_executor",
            factory=FactoryArgs(dtype="bfloat16"),
            engine=EngineArgs(kv_cache_bytes=SPEECH_KV_CACHE_BYTES),
            gpu=0,
            gpu_memory_fraction=0.30,
            next="vocoder",
            stream_to=["vocoder"],
        ),
    ]


class AuraOmniAVPipelineConfig(PipelineConfig):
    """Speech, video and text in; AURA's answer out as text and Qwen3-TTS speech."""

    architecture: ClassVar[str] = "Qwen3VLForConditionalGeneration"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "asr": EngineStageConfig,
        "aura": EngineStageConfig,
        "tts_engine": EngineStageConfig,
    }

    model_path: str
    asr_model_path: str = "Qwen/Qwen3-ASR-1.7B"
    tts_model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    entry_stage: str = "preprocessing"
    terminal_stages_fn: str | None = f"{ROUTING}.resolve_terminal_stages"
    stages: list[StageConfig] = Field(default_factory=aura_omni_av_stages)

    def stage_factory_kwargs(self, stage_name: str) -> AuraOmniStageFactoryKwargs:
        if stage_name == "asr":
            return {"model_path": self.asr_model_path}
        elif stage_name == "vocoder" and (
            not current_platform.enable_tts_vocoder_fast_path()
        ):
            # The vocoder captures its decode graphs during the asynchronous decode warmup, which such a platform skips.
            return {"model_path": self.tts_model_path, "async_decode": False}
        elif stage_name in TTS_STAGES:
            return {"model_path": self.tts_model_path}
        else:
            return {}


EntryClass = AuraOmniAVPipelineConfig
