import numpy as np
import os
import librosa
from datasets import load_dataset
from tqdm import tqdm
from scipy.io import wavfile
import json


def commonvoice_preprocess():
    cache_dir = "_cache/CommonVoice"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(f"{cache_dir}/wav", exist_ok=True)
    os.makedirs(f"{cache_dir}/text", exist_ok=True)

    src_dataset = load_dataset(
        "mozilla-foundation/common_voice_16_1", "en",
        split="test",
        streaming=True,
        trust_remote_code=True,
        cache_dir=cache_dir
    )

    data_info = []
    for idx, instance in tqdm(enumerate(src_dataset)):
        audio = instance.pop("audio")
        wav = librosa.resample(
            audio["array"],
            orig_sr=src_dataset.features["audio"].sampling_rate,
            target_sr=16000
        )
        
        basename = f"{idx:07d}"
        wavfile.write(f"{cache_dir}/wav/{basename}.wav", 16000, (wav * 32767).astype(np.int16))
        with open(f"{cache_dir}/text/{basename}.txt", "w", encoding="utf-8") as f:
            f.write(instance["sentence"])
        data_info.append({
            "basename": basename,
            "length": len(wav),
            "text": instance["sentence"],
            "metadata": instance
        })
    with open(f"{cache_dir}/data_info.json", "w", encoding="utf-8") as f:
        json.dump(data_info, f, indent=4)


def commonvoice_accent():
    cache_dir = "_cache/CommonVoice"
    with open(f"{cache_dir}/data_info.json", "r", encoding="utf-8") as f:
        data_info = json.load(f)
    
    root = "preprocess/cv-accent-split/test"
    for accent in os.listdir(root):
        accent_list = []
        with open(f"{root}/{accent}/list_of_utterances.txt", encoding="utf-8") as f:
            for line in f:
                if line == "\n":
                    continue
                [client_id, filename] = line.strip().split('-')
                accent_list.append((client_id, filename))
        sub = []
        for q in data_info:
            client_id = q["metadata"]["client_id"]
            filename = q["metadata"]["path"].split('/')[-1][:-4]
            if (client_id, filename) in accent_list:
                sub.append(q["basename"])
        print(len(sub))


if __name__ == "__main__":
    commonvoice_preprocess()
    commonvoice_accent()
