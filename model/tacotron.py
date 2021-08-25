from math import sqrt
import random
import torch
from pytorch_lightning import LightningModule

from .layers import AdversarialClassifier
from utils.utils import get_mask_from_lengths
from text import symbols

from utils.plot import plot_alignment_to_numpy, plot_spectrogram_to_numpy, plot_gate_outputs_to_numpy


class Tacotron2(LightningModule):
    def __init__(self, encoder: LightningModule, decoder: LightningModule, 
                 postnet: LightningModule, mask_padding: bool=True,
                 n_mel_channels: int=80, n_frames_per_step: int=3,
                 symbols_lang: str="en", symbols_embedding_dim: int=512,
                 freeze_text: bool=False, load_pretrained_text: str=None,
                 multi_speaker: bool=False, spker_embedding_dim: int=64,
                 n_spker: int=1, speaker_loss_weight: float=0.02,
                 speaker_classifier: AdversarialClassifier=None):

        super(Tacotron2, self).__init__()

        self.save_hyperparameters()

        # init embedding table and text encoder
        if load_pretrained_text != None:
            _pretrained_taco = Tacotron2.load_from_checkpoint(load_pretrained_text, map_location="cpu")
            self.encoder = _pretrained_taco.encoder.to(self.device)
            self.embedding = _pretrained_taco.embedding.to(self.device)
            assert self.encoder.device == self.device == self.embedding.weight.device
            del encoder
            del _pretrained_taco
        else:
            n_symbols = len(symbols(symbols_lang))
            self.embedding = torch.nn.Embedding(
                n_symbols, symbols_embedding_dim)
            std = sqrt(2.0 / (n_symbols + symbols_embedding_dim))
            val = sqrt(3.0) * std  # uniform bounds for std
            self.embedding.weight.data.uniform_(-val, val)

            self.encoder = encoder

        if freeze_text:
            self.embedding.weight.requires_grad = False
            self.encoder.freeze()

        if multi_speaker:
            self.speaker_embedding = torch.nn.Embedding(
                n_spker, spker_embedding_dim)
            std = sqrt(2.0 / (n_spker + spker_embedding_dim))
            val = sqrt(3.0) * std  # uniform bounds for std
            self.speaker_embedding.weight.data.uniform_(-val, val)
            self.speaker_classifier = speaker_classifier
            
        self.decoder = decoder
        self.postnet = postnet

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, spkerID = batch

        return ((text_padded, input_lengths, mel_padded, output_lengths, spkerID),
                (mel_padded, gate_padded, spkerID))

    def parse_output(self, outputs, output_lengths=None):
        if self.hparams.mask_padding and output_lengths is not None:
            output_total_length = outputs[0].size(2)
            mask = ~get_mask_from_lengths(output_lengths, output_total_length)
            mask = mask.expand(self.hparams.n_mel_channels, mask.size(0), mask.size(1))
            if mask.size(2)%self.hparams.n_frames_per_step != 0 :
                to_append = torch.ones( mask.size(0), mask.size(1), (self.hparams.n_frames_per_step-mask.size(2)%self.hparams.n_frames_per_step) ).bool()
                mask = torch.cat([mask, to_append], dim=-1)
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0) # mel_outputs
            outputs[1].data.masked_fill_(mask, 0.0) # mel_outputs_postnet
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs
    
    def compute_loss(self, model_output, targets):
        for t in targets:
            t.requires_grad = False
        
        mel_target, gate_target = targets[0], targets[1]
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _align, speaker_predict = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = torch.nn.MSELoss()(mel_out, mel_target) + \
            torch.nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = torch.nn.BCEWithLogitsLoss()(gate_out, gate_target)

        if speaker_predict != None:
            speaker_predict = speaker_predict.transpose(1, 2) # (B, T, n_speaker) -> (B, n_speaker, T)
            speaker_target = targets[2].unsqueeze(1).repeat(1, speaker_predict.size(2)) # (B) -> (B, T)
            speaker_loss = torch.nn.CrossEntropyLoss()(speaker_predict, speaker_target)
            speaker_loss = speaker_loss * self.hparams.speaker_loss_weight
        else:
            speaker_loss = 0.

        return mel_loss + gate_loss + speaker_loss


    def forward(self, inputs):
        text_inputs, text_lengths, mels, output_lengths, speaker_ids = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        if self.hparams.multi_speaker:
            speaker_outputs = self.speaker_classifier(encoder_outputs) if self.speaker_classifier != None else None
            speaker_embedded = self.speaker_embedding(speaker_ids)[:, None]
            speaker_embedded = speaker_embedded.repeat(1, encoder_outputs.size(1), 1)
            encoder_outputs = torch.cat((encoder_outputs, speaker_embedded), dim=2)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, speaker_outputs],
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


    def log_validation(self, reduced_loss, y, y_pred, iteration):
        self.log("validation.loss", reduced_loss)
        _mel_outputs, mel_outputs, gate_outputs, alignments, *_ = y_pred
        mel_targets, gate_targets, *_ = y

        # plot distribution of parameters
        for tag, value in self.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.experiment.add_histogram(tag, value, iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.logger.experiment.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.logger.experiment.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.logger.experiment.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.logger.experiment.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')

    def training_step(self, batch, batch_idx):
        x, y = self.parse_batch(batch)
        y_pred = self(x)
        loss = self.compute_loss(y_pred, y)
        self.log("train.loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.parse_batch(batch)
        y_pred = self(x)
        loss = self.compute_loss(y_pred, y)
        self.log_validation(loss, y, y_pred, self.global_step)
        return loss