import os
import torch
import torch.nn as nn
import fairseq

from .mos_fairseq import MosPredictor


def download(root):
    ## 1. download the base model from fairseq
    if not os.path.exists(f'{root}/fairseq/wav2vec_small.pt'):
        os.system(f'mkdir -p {root}/fairseq')
        os.system(f'wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt -P {root}/fairseq')
        os.system(f'wget https://raw.githubusercontent.com/pytorch/fairseq/main/LICENSE -P {root}/fairseq')

    ## 2. download the finetuned checkpoint
    if not os.path.exists(f'{root}/pretrained/ckpt_w2vsmall'):
        os.system(f'mkdir -p {root}/pretrained')
        os.system('wget https://zenodo.org/record/6785056/files/ckpt_w2vsmall.tar.gz')
        os.system('tar -zxvf ckpt_w2vsmall.tar.gz')
        os.system(f'mv ckpt_w2vsmall {root}/pretrained/')
        os.system('rm ckpt_w2vsmall.tar.gz')
        os.system(f'cp {root}/fairseq/LICENSE {root}/pretrained/')


def load_mos_model():
    root=os.path.dirname(__file__)
    download(root)
    cp_path = f"{root}/fairseq/wav2vec_small.pt"
    my_checkpoint = f"{root}/pretrained/ckpt_w2vsmall"
    ssl_model_type = 'wav2vec_small.pt'

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ssl_model_type == 'wav2vec_small.pt':
        SSL_OUT_DIM = 768
    elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt']:
        SSL_OUT_DIM = 1024
    else:
        print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.')
        exit()

    model = MosPredictor(ssl_model, SSL_OUT_DIM).to(device)
    model.eval()

    model.load_state_dict(torch.load(my_checkpoint))
    return model
