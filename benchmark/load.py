from torch.utils.data import Dataset
from typing import Type

from .tasks import (
    chime, esd, l2arctic, synth, librispeech
)
from .strategies import (
    basic, dynamic, other
)


STRATEGY = {
    "none": basic.NoStrategy,
    "suta": basic.SUTAStrategy,
    "csuta": basic.CSUTAStrategy,

    "dsuta": dynamic.DSUTAStrategy,
    "dcsuta": dynamic.DCSUTAStrategy,
    "trans": dynamic.TranscriptionStrategy,

    "csuta*": other.CSUTAResetStrategy,
    "multiexpert": other.MultiExpertStrategy,
    "multiexpert_trans": other.MultiExpertTransStrategy,
    "advanced": other.AdvancedStrategy,
}


TASK = {
    "synth_random": synth.RandomSequence,
    "synth_content": synth.ContentSequence,
    "synth_speaker": synth.SpeakerSequence,
    "synth_noise": synth.NoiseSequence,
    "esd_random": esd.RandomSequence,
    "esd_speaker": esd.SpeakerSequence,
    "esd_emotion": esd.EmotionSequence,
    "esd_content": esd.ContentSequence,
    "l2arctic_random": l2arctic.RandomSequence,
    "l2arctic_accent": l2arctic.AccentSequence,
    "l2arctic_content": l2arctic.ContentSequence,
    "l2arctic_speaker": l2arctic.SpeakerSequence,
    "chime_random": chime.RandomSequence,
    "librispeech_random": librispeech.RandomSequence,
}


def get_strategy(name) -> Type[basic.BaseStrategy]:
    return STRATEGY[name]


def get_task(name) -> Dataset:
    return TASK[name]()
