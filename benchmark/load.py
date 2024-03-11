from torch.utils.data import Dataset
from typing import Type

from .tasks import chime, esd, l2arctic, synth
from .basic import BaseStrategy, NoStrategy, SUTAStrategy, CSUTAStrategy
from .loss_based import *
from . import dynamic
from . import other


STRATEGY = {
    "none": NoStrategy,
    "suta": SUTAStrategy,
    "csuta": CSUTAStrategy,

    "dsuta": dynamic.DSUTAStrategy,
    "dcsuta": dynamic.DCSUTAStrategy,
    "trans": dynamic.TranscriptionStrategy,
    
    "csuta*": other.CSUTAResetStrategy,
    "multiexpert": other.MultiExpertStrategy,
    "multiexpert-trans": other.MultiExpertTransStrategy,
    "advanced": other.AdvancedStrategy,
}


TASK = {
    "task1": synth.RandomSequence,
    "task2": synth.ContentSequence,
    "task3": synth.SpeakerSequence,
    "task4": synth.NoiseSequence,
    "task5": esd.RandomSequence,
    "task6": esd.SpeakerSequence,
    "task7": esd.EmotionSequence,
    "task8": esd.ContentSequence,
    "task9": l2arctic.RandomSequence,
    "task10": l2arctic.AccentSequence,
    "task11": l2arctic.ContentSequence,
    "task12": l2arctic.SpeakerSequence,
    "task13": chime.RandomSequence
}


def get_strategy(name) -> Type[BaseStrategy]:
    return STRATEGY[name]


def get_task(name) -> Dataset:
    return TASK[name]()
