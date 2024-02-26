import numpy as np
import torch
import json
import librosa
from datasets import load_dataset
from tqdm import tqdm
import re
from builtins import str as unicode
from scipy.io import wavfile


class SutaCorpus(object):
    def __init__(self) -> None:
        self.root = "/mnt/d/Projects/tts_api/suta"
        with open(f"{self.root}/metadata.json", 'r') as f:
            self.metadata = json.load(f)

        # load noise
        self.musan_root = "/mnt/d/Data/musan"
        background_noise_filename = [
            "noise-free-sound-0232",
            "noise-free-sound-0840",
            "noise-free-sound-0841",
            "noise-free-sound-0842",
            "noise-free-sound-0671",
            "noise-free-sound-0234",
            "noise-free-sound-0248",
            "noise-free-sound-0030",
            "noise-free-sound-0031",
            "noise-free-sound-0050",
        ]
        self.musan_noise_paths = [
            f"{self.musan_root}/noise/free-sound/{x}.wav" for x in background_noise_filename
        ]

    def __len__(self):
        return 1000
    
    def get(self, content_idx, speaker_idx, noise_idx) -> np.ndarray:
        basename = f"{content_idx * 10 + speaker_idx:03d}"
        wav_path = f"{self.root}/wavs/{basename}.wav"
        wav, _ = librosa.load(wav_path, sr=16000)
        text = self.metadata[basename]["text"]

        # add musan noise
        noise, _  = librosa.load(self.musan_noise_paths[noise_idx], sr=16000)
        if len(noise) > len(wav):
            noise = noise[:len(wav)]
        wav[:len(noise)] +=  noise * 0.2
        # wavfile.write("check.wav", 16000, (wav * 32767).astype(np.int16))
        # input()

        return {
            "wav": wav,
            "text": text
        }


class LibriSpeechCorpus(object):
    def __init__(self, extra_noise=0.0) -> None:
        self.src_dataset = load_dataset(
            "librispeech_asr",
            split="test.other",
            streaming=True
        )

        # create gaussian noise
        self.extra_noise = extra_noise
        self.noises = []
        self.instances = []
        for instance in tqdm(self.src_dataset):
            wav = instance["audio"]["array"]
            self.noises.append(np.random.randn(*wav.shape))
            self.instances.append({
                "audio": wav,
                "text": instance["text"]
            })

    def __len__(self):
        return len(self.noises)
    
    def get(self, idx) -> np.ndarray:
        instance = self.instances[idx]
        wav = instance["audio"]
        text = instance["text"]
        noise = self.noises[idx]

        return {
            "wav": wav + self.extra_noise * noise,
            "text": text
        }


class CommonVoiceCorpus(object):
    def __init__(self) -> None:
        self.src_dataset = load_dataset(
            "mozilla-foundation/common_voice_16_1",
            "en",
            split="test",
            streaming=True
        )

        self.instances = []
        for instance in tqdm(self.src_dataset):
            wav = librosa.resample(
                instance["audio"]["array"],
                orig_sr=self.src_dataset.features["audio"].sampling_rate,
                target_sr=16000
            )
            self.instances.append({
                "audio": wav,
                "text": instance["sentence"]
            })
            # wavfile.write("check.wav", 16000, (wav * 32767).astype(np.int16))
            # input()

    def __len__(self):
        return len(self.instances)
    
    def get(self, idx) -> np.ndarray:
        instance = self.instances[idx]
        wav = instance["audio"]
        text = preprocess_text(instance["text"])

        return {
            "wav": wav,
            "text": text
        }


def preprocess_text(text):
    text = unicode(text)
    text = text.replace("i.e.", "that is")
    text = text.replace("e.g.", "for example")
    text = text.replace("Mr.", "Mister")
    text = text.replace("Mrs.", "Mistress")
    text = text.replace("Dr.", "Doctor")
    text = text.replace("-", " ")
    text = text.upper()
    text = re.sub("[^ A-Z']", "", text)
    text = ' '.join(text.split())
    
    return text
