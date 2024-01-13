import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from copy import deepcopy

from main import collect_params, setup_optimizer
from main import softmax_entropy, mcc_loss, div_loss


class SUTASystem(object):

    SAMPLE_RATE = 16000

    def __init__(self, args) -> None:
        self.history = {}

        # load model and tokenizer
        self.processor = Wav2Vec2Processor.from_pretrained(args.asr, sampling_rate=SUTASystem.SAMPLE_RATE, return_attention_mask=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(args.asr).eval().cuda()

        # set up for tent
        self.optimizer, self.scheduler = setup_optimizer(self.build_optimized_model(args), args.opt, args.lr, scheduler=args.scheduler)

    def build_optimized_model(self, args):
        self.model.requires_grad_(False)
        params, param_names = collect_params(self.model, args.bias_only, args.train_feature, args.train_all, args.train_LN)
        # print(param_names)
        for p in params:
            p.requires_grad = True
        return params

    def _wav_to_model_input(self, wavs) -> torch.FloatTensor:
        inputs = self.processor(wavs, sampling_rate=SUTASystem.SAMPLE_RATE, return_tensors="pt", padding="longest")
        return inputs.input_values.to(self.model.device)

    def adapt(self, wavs, em_coef=0.9, reweight=False, temp=1., not_blank=True, 
                        div_coef=0, repeat_inference=True, skip_short_thd=None):
        """Forward and adapt model on batch of data.

        Measure entropy of the model prediction, take gradients, and update params.

        the index of <pad> in vocab is 0
        """
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

        if 1 - em_coef > 0: 
            c_loss = mcc_loss(outputs / temp, reweight)
            loss += c_loss * (1 - em_coef)

        if div_coef > 0: 
            d_loss = div_loss(outputs, not_blank) 
            loss += d_loss * div_coef 

        loss.backward()
        # print(loss)
        self.optimizer.step()
        if self.scheduler is not None: 
            self.scheduler.step()
        self.model.zero_grad()

    @torch.no_grad()
    def inference(self, wavs):
        x = self._wav_to_model_input(wavs)
        outputs = self.model(x).logits
        predicted_ids = torch.argmax(outputs, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        
        return list(transcription)

    def snapshot(self, key: str):
        """Copy the model and optimizer states for resetting after adaptation."""
        print(f"Store state. (key: {key})")
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        if self.scheduler is not None:
            scheduler_state = deepcopy(self.scheduler.state_dict())
        else:
            scheduler_state = None
        self.history[key] = (model_state, optimizer_state, scheduler_state)
    
    def load_snapshot(self, key: str) -> None:
        """Restore the model and optimizer states from copies."""
        print(f"Reset. (key: {key})")
        model_state, optimizer_state, scheduler_state = self.history[key]
        model_state = deepcopy(model_state)
        optimizer_state = deepcopy(optimizer_state)
        self.model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(optimizer_state)
        if self.scheduler is not None:
            scheduler_state = deepcopy(scheduler_state)
            self.scheduler.load_state_dict(scheduler_state)
