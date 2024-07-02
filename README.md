# Listen, Adapt, Better WER: Source-free Single-utterance Test-time Adaptation for Automatic Speech Recognition 
![](https://i.imgur.com/pPAS730.png)
### Introduction
Given a CTC-based trained ASR model, we proposed **Single-Utterance Test-time Adaptation (SUTA)** framework, which can adapt the source ASR model for one utterance by unsupervised objectives (such as entropy minimization, minimum class confusion). For details of SUTA's method and experimental results, please see our paper [[link]](https://arxiv.org/abs/2203.14222).

Our proposed SUTA has below advantages:
* Efficient adaptation for one **single utterance**
* Don't need to access the source data
* About 0.1s per 1s utterance for 10-steps adaptation

### Installation 
```
pip install -r requirements.txt
```
Local install (pip install -e) [another repo](https://github.com/hhhaaahhhaa/dlhlp-lib.git).

### Data Preparation
Currently, our code only supports [Librispeech](https://www.openslr.org/12)/[CHiME-3](https://catalog.ldc.upenn.edu/LDC2017S24)/[Common voice En](https://tinyurl.com/cvjune2020)/[TED-LIUM 3](https://www.openslr.org/51/)
You have to download datasets by your own. After downloading, set your local paths in ```corpus/Define.py```.

### Usage
The source ASR model is [w2v2-base fine-tuned on Librispeech 960 hours](https://huggingface.co/facebook/wav2vec2-base-960h). The pre-trained model is imported by Huggingface.

Run SUTA on different datasets: (Removed)
```
bash scripts/{dataset_name: LS/CH/CV/TD}.sh
```
### Results
|                     | LS+0 | LS+0.005 | LS+0.01 | CH   | CV   | TD   |
|---------------------|------|----------|---------|------|------|------|
| Source ASR model    | 8.6  | 13.9     | 24.4    | 31.2 | 36.8 | 13.2 |
| Baseline TTA - SDPL | 8.3  | 13.1     | 23.1    | 30.4 | 36.3 | 12.8 |
| Proposed TTA - SUTA | 7.3  | 10.9     | 16.7    | 25.0 | 31.2 | 11.9 |

### Benchmark
View all definition of tasks and strategies in ```benchmark/load.py```. Run the following command for example
```python
python run_benchmark.py -s suta -t synth_random
```

### TODO 
- Dynamic step research
  - loss curve
  - LM scoring
- Continual TTA research
  - CSUTA strategies with batch update


### Contact 
* Guan-Ting Lin [email] daniel094144@gmail.com
* Wei-Ping Huang [email] thomas1232121@gmail.com
### Citation
```
@article{lin2022listen,
  title={Listen, Adapt, Better WER: Source-free Single-utterance Test-time Adaptation for Automatic Speech Recognition},
  author={Lin, Guan-Ting and Li, Shang-Wen and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2203.14222},
  year={2022}
}
```

