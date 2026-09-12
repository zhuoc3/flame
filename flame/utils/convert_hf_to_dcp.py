# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP
from transformers import AutoModelForCausalLM

# Add powerdata to path for powerformer_hf / powerssm
_powerdata_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_powerdata_dir))

import fla  # noqa
# Model registrations: side-effect-only imports for the Auto* classes. A model
# absent from this checkout must not take training down with it -- powergdn
# lives on its own branch and is not present on powerdelta, and as a bare
# import it crash-looped every member of a live sweep at seed time
# (ModuleNotFoundError, 7 chained retries under --dependency=afterany).
#
# Nothing is masked that matters: a model actually being trained still fails
# loudly a few lines later, when AutoConfig raises "unrecognized model type".
# `logger` is not defined this early in the module, hence stderr.
for _registration in ("powerformer_hf", "powerssm", "powerdelta", "powergdn"):
    try:
        __import__(_registration)
    except ImportError as _exc:
        print(
            f"[flame] model registration skipped: {_registration} ({_exc})",
            file=sys.stderr,
        )
from torchtitan.tools.logging import init_logger, logger


@torch.inference_mode()
def convert_hf_weights(model: str, checkpoint: str):
    logger.info(f"Loading model from {model}")
    model = AutoModelForCausalLM.from_pretrained(model)
    state_dict = model.state_dict()

    logger.info(f"Writing to DCP at '{checkpoint}'")
    checkpoint.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)
    DCP.save({"model": state_dict}, storage_writer=storage_writer)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(description="Convert huggingface-style model weights to DCP format.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    convert_hf_weights(args.model, args.checkpoint)
