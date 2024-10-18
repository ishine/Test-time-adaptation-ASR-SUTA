import os
import typing

from ..utils.load import get_class_in_module
from .base import IStrategy


SRC_DIR = "src/strategies"

BASIC = {
    "none": (f"{SRC_DIR}/basic.py", "NoStrategy"),
    "suta": (f"{SRC_DIR}/basic.py", "SUTAStrategy"),
    "csuta": (f"{SRC_DIR}/basic.py", "CSUTAStrategy"),
    "sdpl": (f"{SRC_DIR}/basic.py", "SDPLStrategy"),
}

DSUTA = {
    "dsuta": (f"{SRC_DIR}/dsuta.py", "DSUTAStrategy"),
    "dsuta-reset": (f"{SRC_DIR}/dsuta_reset.py", "DSUTAResetStrategy"),
}

EC = {
    "rescore": (f"{SRC_DIR}/error_correction.py", "RescoreStrategy"),
    "LLM": (f"{SRC_DIR}/error_correction.py", "LLMStrategy"),
    "aLLM": (f"{SRC_DIR}/error_correction.py", "AsyncLLMStrategy"),
}

OTHER = {
    "awmc": (f"{SRC_DIR}/awmc.py", "AWMCStrategy"),
}

EXP = {
    "overfit": (f"{SRC_DIR}/upperbound.py", "OverfitStrategy"),
}

STRATEGY_MAPPING = {
    **BASIC,
    **DSUTA,
    **EC,
    **OTHER,
    **EXP,
}


def get_strategy_cls(name) -> typing.Type[IStrategy]:
    module_path, class_name = STRATEGY_MAPPING[name]
    return get_class_in_module(class_name, module_path)
