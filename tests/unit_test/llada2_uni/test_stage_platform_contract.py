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

from types import SimpleNamespace
from typing import Any

import pytest

import sglang_omni.platforms as platforms
from sglang_omni.models.llada2_uni import stages
from sglang_omni.models.llada2_uni.config import IMAGE_STAGE, THINKER_STAGE, EntryClass


class _FakePlatform:
    """A platform stand-in carrying only the hooks these stages consult."""

    device_type = "xpu"

    def __init__(self, *, dllm_backend: str | None, dllm_graph: bool) -> None:
        self._dllm_backend = dllm_backend
        self._dllm_graph = dllm_graph

    def get_dllm_attention_backend(self) -> str | None:
        return self._dllm_backend

    def enable_dllm_decode_graph(self) -> bool:
        return self._dllm_graph


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
        # Mirrors a freshly built ServerArgs: cuda_graph_config is still the
        # unresolved None here (see the regression test below), so a stand-in
        # that fills it in would hide a crash on the real object.
        return SimpleNamespace(
            attention_backend=kwargs.get("attention_backend"),
            dllm_algorithm=kwargs.get("dllm_algorithm"),
            mem_fraction_static=None,
            disable_cuda_graph=bool(kwargs.get("disable_cuda_graph", False)),
            cuda_graph_config=None,
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


def test_the_thinker_only_reads_server_args_that_are_resolved_by_then(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The factory must survive an unresolved ``cuda_graph_config``.

    SGLang folds the graph flags into that field in its resolution pass, which
    runs inside the engine bootstrap this factory calls last, so a freshly built
    ServerArgs still carries the dataclass default -- and a ``resolved_view``
    read taken before that pass answers with the same raw input, since a
    declaration-only resolver never writes the field. Reading it beforehand
    killed the stage at construction -- ``AttributeError: 'NoneType' object has
    no attribute 'decode'`` -- before any weight was touched, on every platform,
    and the stand-in used here is what let it through.
    """
    import dataclasses

    from sglang.srt.server_args import ServerArgs

    field = {f.name: f for f in dataclasses.fields(ServerArgs)}["cuda_graph_config"]
    assert field.default is None, "stub below must mirror the unresolved default"

    _, scheduler = _drive_thinker(
        monkeypatch, _FakePlatform(dllm_backend="triton", dllm_graph=False)
    )
    assert scheduler["server_args"].cuda_graph_config is None


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
