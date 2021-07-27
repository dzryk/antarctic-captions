import torch
import pytorch_lightning as pl

from CLIP import clip
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import dataset
import utils


class Model(pl.LightningModule):
    """
    Model consists of CLIP with an Encoder-Decoder model.
    Uses BART-Large by default. Only layernorm params for the
    Encoder-Decoder are learned.
    """
    def __init__(self, args=None, cache=None, cache_emb=None):
        super(Model, self).__init__()
        self.save_hyperparameters(args, ignore='cache_emb')
        self.perceiver = clip.load(self.hparams.clip_model, jit=False)[0]
        self.tokenizer = AutoTokenizer.from_pretrained(
            'facebook/bart-large', cache_dir=self.hparams.datadir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            'facebook/bart-large', cache_dir=self.hparams.datadir)
        self.cache = cache
        self.cache_emb = cache_emb
        
        # Shut off most gradients
        for pname, p in self.model.named_parameters():
            pname = pname.lower()
            if 'layer_norm' in pname and 'encoder' in pname:
                p.requires_grad = True
            else:
                p.requires_grad = False
        for pname, p in self.perceiver.named_parameters():
            p.requires_grad = False
    
    def compute_loss(self, x, y):
        cache_emb = torch.tensor(self.cache_emb, device=self.device)
        x = utils.build_table(x, 
                              perceiver=self.perceiver,
                              cache=self.cache,
                              cache_emb=cache_emb,
                              topk=self.hparams.topk)

        # Uses a different caption per image depending on epoch.
        y = y[self.current_epoch % self.hparams.ncaptions]
        
        enc_inputs = self.tokenizer(x,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.hparams.maxlen_enc,
                                    return_tensors='pt')
        dec_inputs = self.tokenizer(y,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.hparams.maxlen_dec,
                                    return_tensors='pt')
        
        # Label entries of -100 are ignored when computing loss.
        # These lines are specific to BART tokenizer.
        labels = -100 * torch.ones((len(x), dec_inputs.input_ids.shape[-1]), dtype=torch.long)
        labels[:,:-1] = -101 * (1 - dec_inputs.attention_mask[:,1:]) + dec_inputs.input_ids[:,1:]
        
        loss = self.model(input_ids=torch.tensor(
                              enc_inputs.input_ids, device=self.device),
                          attention_mask=torch.tensor(
                              enc_inputs.attention_mask, device=self.device),
                          decoder_input_ids=torch.tensor(
                              dec_inputs.input_ids, device=self.device),
                          decoder_attention_mask=torch.tensor(
                              dec_inputs.attention_mask, device=self.device),
                          labels=torch.tensor(labels, device=self.device)).loss
        return loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.compute_loss(x, y)
        self.log('loss', loss, on_step=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.compute_loss(x, y)
        self.log('vloss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lrate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.hparams.tmax)
        return [optimizer], [scheduler]