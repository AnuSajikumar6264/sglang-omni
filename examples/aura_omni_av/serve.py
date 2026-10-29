# SPDX-License-Identifier: Apache-2.0
"""Serve AURA Omni with its spoken answer merged into non-streaming chat replies.

    python examples/aura_omni_av/serve.py --config examples/configs/aura_omni_av_xpu_b60.yaml --port 8000

Dotted stage flags work as with sgl-omni serve, for example --aura.gpu 3 or
--tts_model_path /path/to/Qwen3-TTS-12Hz-1.7B-Base.
"""

from __future__ import annotations

import argparse
import logging

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.aura_omni_av.request_builders import build_chat_result
from sglang_omni.serve.launcher import launch_server
from sglang_omni.utils.logging import configure_dependency_loggers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Path to a pipeline config file.")
    source.add_argument("--model-path", help="AURA checkpoint, a repo id or a path.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--allowed-local-media-path", default=None)
    args, stage_flags = parser.parse_known_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    configure_dependency_loggers()
    if args.config is not None:
        config_manager = ConfigManager.from_file(args.config)
    else:
        config_manager = ConfigManager.from_model_path(args.model_path)
    pipeline_config = config_manager.merge_config(
        config_manager.parse_extra_args(stage_flags)
    )
    launch_server(
        pipeline_config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        log_level=args.log_level,
        client_kwargs={"result_builder": build_chat_result},
        allowed_local_media_path=args.allowed_local_media_path,
    )


if __name__ == "__main__":
    main()
