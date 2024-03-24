import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from copy import deepcopy
import json

from .loss import softmax_entropy, mcc_loss, div_loss
from utils.tool import pad_1D


class SUTASystem(object):

    SAMPLE_RATE = 16000

    def __init__(self, config) -> None:
        self.config = config
        self.history = {}
        self.adapt_count = 0

        # load model and tokenizer
        self.processor = Wav2Vec2Processor.from_pretrained(config["model_name"], sampling_rate=SUTASystem.SAMPLE_RATE, return_attention_mask=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(config["model_name"]).eval().cuda()

        # set up for tent
        self.optimizer, self.scheduler = setup_optimizer(
            self.build_optimized_model(),
            config["opt"], config["lr"], scheduler=config["scheduler"]
        )

        # for PL loss
        f = open('vocab.json')
        self.vocab = json.load(f)
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=False)

    def build_optimized_model(self):
        self.model.requires_grad_(False)
        params, param_names = self.collect_params()
        # print(param_names[:10])
        for p in params:
            p.requires_grad = True
        print("Optimizable: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        return params

    def _wav_to_model_input(self, wavs) -> torch.FloatTensor:
        inputs = self.processor(wavs, sampling_rate=SUTASystem.SAMPLE_RATE, return_tensors="pt", padding="longest")
        return inputs.input_values.to(self.model.device)

    def reset_adapt_counter(self):
        self.adapt_count = 0
    
    def l2_loss(self):
        l2_loss = 0.0
        assert "init" in self.history
        orig_state_dict = self.history["init"][0]

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                l2_loss += torch.sum((param - orig_state_dict[name]) ** 2)
        return l2_loss

    def pseudo_labeling_loss(self, outputs, transcriptions):
        targets = []
        for transcription in transcriptions:
            target = []
            for s in transcription:
                if s == ' ':
                    s = '|'
                target.append(self.vocab[s])
            targets.append(np.array(target))
        tgt_lens = [len(t) for t in targets]
        # print(targets)
        targets = torch.from_numpy(pad_1D(targets)).int()

        logp = outputs.log_softmax(2).transpose(1, 0) # L,N,D
        input_len = [logp.shape[0]] * len(tgt_lens)
        loss = self.ctc_loss(logp, targets, torch.tensor(input_len), torch.tensor(tgt_lens))
        
        return loss

    def adapt(self, wavs, em_coef=0.9, reweight=False, temp=1., not_blank=True, 
                        div_coef=0, l2_coef=0, repeat_inference=True, skip_short_thd=None, record={}):
        """Forward and adapt model on batch of data.

        Measure entropy of the model prediction, take gradients, and update params.

        the index of <pad> in vocab is 0
        """
        self.adapt_count += 1
        x = self._wav_to_model_input(wavs)

        # forward
        outputs = self.model(x).logits
        
        predicted_ids = torch.argmax(outputs, dim=-1)
        non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
        # adapt
        loss = 0

        if em_coef > 0: 
            if not_blank:      
                e_loss = softmax_entropy(outputs / temp)[non_blank].mean(0).mean()
        
            else: 
                e_loss = softmax_entropy(outputs / temp).mean(0).mean() 
            
            loss += e_loss * em_coef
            record["e_loss"] = e_loss.item()

        if 1 - em_coef > 0: 
            c_loss = mcc_loss(outputs / temp, reweight)
            loss += c_loss * (1 - em_coef)
            record["c_loss"] = c_loss.item()

        if div_coef > 0: 
            d_loss = div_loss(outputs, not_blank) 
            loss += d_loss * div_coef
            record["d_loss"] = d_loss.item()
        
        record["total_loss"] = loss.item()

        if l2_coef > 0: 
            l2_loss = self.l2_loss()
            loss += l2_loss * l2_coef
            record["l2_loss"] = l2_loss.item()

        self.model.zero_grad()
        loss.backward()
        # print(e_loss.item(), c_loss.item(), l2_loss.item())
        # print(predicted_ids)
        self.optimizer.step()
        if self.scheduler is not None: 
            self.scheduler.step()
        self.model.zero_grad()
        if torch.isnan(e_loss):
            return False
        return True

    def pl_adapt(self, wavs, transcriptions: list[str]):
        # forward
        self.adapt_count += 1
        x = self._wav_to_model_input(wavs)
        outputs = self.model(x).logits
        predicted_ids = torch.argmax(outputs, dim=-1)
        non_blank = torch.where(predicted_ids != 0, 1, 0).bool()

        # adapt
        loss = self.pseudo_labeling_loss(outputs, transcriptions)

        self.model.zero_grad()
        loss.backward()
        # grad = cal_grad(model)
        # print(grad)
        self.optimizer.step()
        if self.scheduler is not None: 
            self.scheduler.step()
        self.model.zero_grad()

        if torch.isnan(loss) or (not torch.any(non_blank)):
            return False
        return True

    @torch.no_grad()
    def inference(self, wavs):
        x = self._wav_to_model_input(wavs)
        outputs = self.model(x).logits
        predicted_ids = torch.argmax(outputs, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        
        return list(transcription)
    
    @torch.no_grad()
    def calc_loss(self, wavs, em_coef=0.9, reweight=False, temp=1., not_blank=True):
        x = self._wav_to_model_input(wavs)
        outputs = self.model(x).logits
        
        predicted_ids = torch.argmax(outputs, dim=-1)
        non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
        loss = 0
        if em_coef > 0: 
            if not_blank:      
                e_loss = softmax_entropy(outputs / temp)[non_blank].mean(0).mean()
        
            else: 
                e_loss = softmax_entropy(outputs / temp).mean(0).mean() 
            
            loss += e_loss * em_coef

        if 1 - em_coef > 0: 
            c_loss = mcc_loss(outputs / temp, reweight)
            loss += c_loss * (1 - em_coef)

        # l2_loss = self.l2_loss() * self.config["l2_coef_crit"]
        # loss += l2_loss
        # print(l2_loss.item())

        return {
            "e_loss": e_loss.item(),
            "c_loss": c_loss.item(),
            # "l2_loss": l2_loss.item(),
            "total_loss": loss.item()
        }

    def snapshot(self, key: str):
        """Copy the model and optimizer states for resetting after adaptation."""
        # print(f"Store state. (key: {key})")
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        if self.scheduler is not None:
            scheduler_state = deepcopy(self.scheduler.state_dict())
        else:
            scheduler_state = None
        self.history[key] = (model_state, optimizer_state, scheduler_state)
    
    def load_snapshot(self, key: str) -> None:
        """Restore the model and optimizer states from copies."""
        # print(f"Reset. (key: {key})")
        model_state, optimizer_state, scheduler_state = self.history[key]
        model_state = deepcopy(model_state)
        self.model.load_state_dict(model_state, strict=True)
        
        if optimizer_state is not None:
            # optimizer_state = self.history["init"][1]
            optimizer_state = deepcopy(optimizer_state)
            self.optimizer.load_state_dict(optimizer_state)
        if scheduler_state is not None:
            scheduler_state = deepcopy(scheduler_state)
            self.scheduler.load_state_dict(scheduler_state)

    def delete_snapshot(self, key: str) -> None:
        """Delete specific history."""
        self.history.pop(key)

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        trainable = []
        if self.config["bias_only"]:
            trainable = ['bias']
        else: 
            trainable = ['weight', 'bias']

        if self.config.get("bitfit", False):
            for np, p in self.model.named_parameters():
                if str(np).split('.')[1] == 'encoder' and "bias" in np:
                    p.requires_grad = True
                    params.append(p)
                    names.append(np)
        
        for nm, m in self.model.named_modules():
            # print(nm)
            if self.config["train_LN"]: 
                if isinstance(m, nn.LayerNorm):
                    for np, p in m.named_parameters():
                        if np in trainable:  
                            p.requires_grad = True
                            params.append(p)
                            names.append(f"{nm}.{np}")
            if self.config["train_feature"]:
                if len(str(nm).split('.')) > 1:
                    if str(nm).split('.')[1] == 'feature_extractor' or str(nm).split('.')[1] == 'feature_projection':
                        for np, p in m.named_parameters():
                            p.requires_grad = True
                            params.append(p)
                            names.append(f"{nm}.{np}")
                            
            if self.config["train_all"]: 
                for np, p in m.named_parameters():
                    p.requires_grad = True
                    params.append(p)
                    names.append(f"{nm}.{np}")

        return params, names


def setup_optimizer(params, opt_name='AdamW', lr=1e-4, beta=0.9, weight_decay=0., scheduler=None, step_size=1, gamma=0.7):
    opt = getattr(torch.optim, opt_name)
    print(f'[INFO]    optimizer: {opt}')
    print(f'[INFO]    scheduler: {scheduler}')
    if opt_name == 'Adam':       
        optimizer = opt(params,
                lr=lr,
                betas=(beta, 0.999),
                weight_decay=weight_decay)
    else: 
        optimizer = opt(params, lr=lr, weight_decay=weight_decay)
    
    if scheduler is not None: 
        return optimizer, eval(scheduler)(optimizer, step_size=step_size, gamma=gamma)
    else: 
        return optimizer, None
