import logging
import os
import sys
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from yolox.utils.torch_utils import ModelEMA
from yolox.utils.loggers import colorstr

__all__ = [
    "ALMOST_ONE",
    # "QAT_BATCH_SCALE",
    # "sparsezoo_download",
    "ToggleableModelEMA",
    "load_ema",
    # "load_sparsified_model",
    # "neuralmagic_onnx_export",
    # "export_sample_inputs_outputs",
]

SAVE_ROOT = Path.cwd()
RANK = int(os.getenv("RANK", -1))
ALMOST_ONE = 1 - 1e-9  # for incrementing epoch to be applied to recipe

class ToggleableModelEMA(ModelEMA):
    """
    Subclasses YOLOv5 ModelEMA to enabled disabling during QAT
    """

    def __init__(self, enabled, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enabled = enabled

    def update(self, *args, **kwargs):
        if self.enabled:
            super().update(*args, **kwargs)


def load_ema(
    ema_state_dict: Dict[str, Any],
    model: torch.nn.Module,
    enabled: bool = True,
    **ema_kwargs,
) -> ToggleableModelEMA:
    """
    Loads a ToggleableModelEMA object from a ModelEMA state dict and loaded model
    """
    ema = ToggleableModelEMA(enabled, model, **ema_kwargs)
    ema.ema.load_state_dict(ema_state_dict)
    return ema

def set_logging(name=None, verbose=True):
    # Sets level and returns logger
    # if is_kaggle() or is_colab():
    #     for h in logging.root.handlers:
    #         logging.root.removeHandler(h)  # remove all handlers associated with the root logger object
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    log.addHandler(handler)


set_logging()  # run before defining LOGGER
LOGGER = logging.getLogger("YOLOX")  # define globally (used in train.py, val.py, detect.py, etc.)

def nm_log_console(message: str, logger: "Logger" = None, level: str = "info"):
    """
    Log sparsification-related messages to the console

    :param message: message to be logged
    :param level: level to be logged at
    """
    # default to global logger if none provided
    logger = logger or LOGGER

    if RANK in [0, -1]:
        if level == "warning":
            logger.warning(
                f"{colorstr('Neural Magic: ')}{colorstr('yellow', 'warning - ')}"
                f"{message}"
            )
        else:  # default to info
            logger.info(f"{colorstr('Neural Magic: ')}{message}")
