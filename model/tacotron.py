from math import sqrt

import torch
from pytorch_lightning import LightningModule

from utils.utils import get_mask_from_lengths
from text import symbols


class Tacotron2(LightningModule):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, 
                 postnet: torch.nn.Module, mask_padding: bool=True,
                 n_mel_channels: int=80, n_frames_per_step: int=3,
                 symbols_lang: str="en", symbols_embedding_dim: int=512,
                 learning_rate: float=1e-3, weight_decay: float=1e-6):

        super(Tacotron2, self).__init__()
        self.automatic_optimization = False

        self.save_hyperparameters("mask_padding", "n_mel_channels", "n_frames_per_step", "symbols_lang", "symbols_embedding_dim")
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        n_symbols = len(symbols(self.hparams.symbols_lang))
        self.embedding = torch.nn.Embedding(
            n_symbols, self.hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (n_symbols + self.hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        self.encoder = encoder
        self.decoder = decoder
        self.postnet = postnet

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch

        return ((text_padded, input_lengths, mel_padded, output_lengths),
                (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            output_total_length = outputs[0].size(2)
            mask = ~get_mask_from_lengths(output_lengths, output_total_length)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            if mask.size(2)%self.n_frames_per_step != 0 :
                to_append = torch.ones( mask.size(0), mask.size(1), (self.n_frames_per_step-mask.size(2)%self.n_frames_per_step) ).bool().to(mask.device)
                mask = torch.cat([mask, to_append], dim=-1)
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0) # mel_outputs
            outputs[1].data.masked_fill_(mask, 0.0) # mel_outputs_postnet
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs
    
    def compute_loss(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = torch.nn.MSELoss()(mel_out, mel_target) + \
            torch.nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = torch.nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss

    def forward(self, inputs):
        text_inputs, text_lengths, mels, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs

    def training_step(self, batch, batch_idx):
        x, y = self.parse_batch(batch)
        y_pred = self(x)
        loss = self.compute_loss(y_pred, y)
        return loss