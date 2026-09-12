# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
import io
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

# Add powerdata to path for powerformer_hf
_powerdata_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_powerdata_dir))

import torch
import torch.serialization
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
def save_pretrained(
    path: str,
    step: int,
    config: str,
    tokenizer: str
):
    logger.info(f"Loading the config from {config}")
    config = AutoConfig.from_pretrained(config, trust_remote_code=True)

    logger.info(f"Saving the config to {path}")
    config.save_pretrained(path)
    logger.info(f"Loading the tokenizer from {tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    logger.info(f"Saving the tokenizer to {path}")
    tokenizer.save_pretrained(path)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = os.path.join(path, f'checkpoint/step-{step}')
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
        logger.info(f"Saving the distributed checkpoint to {checkpoint_path}")
        dcp_to_torch_save(checkpoint, checkpoint_path)

        logger.info(f"Initializing the model from config\n{config}")
        model = AutoModelForCausalLM.from_config(config)
        logger.info(model)
        logger.info("Loading state dict from the checkpoint")

        # Add datetime.timedelta and io.BytesIO to safe globals
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        # torch.load now with default weights_only=True will work
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model'])

        logger.info(f"Saving the model to {path}")
        model.save_pretrained(path)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser("Convert DCP format model weights to huggingface-style.")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    args = parser.parse_args()
    save_pretrained(args.path, args.step, args.config, args.tokenizer)
