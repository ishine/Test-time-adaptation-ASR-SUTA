from torch.utils.data import Dataset
from typing import Type

from .task import *
from .basic import BaseStrategy, NoStrategy, SUTAStrategy, CSUTAStrategy
from .loss_based import *
from .other import *


STRATEGY = {
    "none": NoStrategy,
    "suta": SUTAStrategy,
    "csuta": CSUTAStrategy,
    "csuta*": CSUTAResetStrategy,
    "multiexpert": MultiExpertStrategy,
}


TASK = {
    "task1": Task1,
    "task2": Task2,
    "task3": Task3,
    "task4": Task4,
    "task5": Task5,
}


def get_strategy(name) -> Type[BaseStrategy]:
    return STRATEGY[name]


def get_task(name) -> Dataset:
    return TASK[name]()
