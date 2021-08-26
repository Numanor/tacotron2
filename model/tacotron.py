from math import sqrt
import random
from typing import Dict
from numpy import mod
import torch
from pytorch_lightning import LightningModule

from .layers import AdversarialClassifier
from utils.utils import get_mask_from_lengths
from text import symbols
from utils.plot import plot_alignment_to_numpy, plot_spectrogram_to_numpy, plot_gate_outputs_to_numpy

from hifigan.vocoder import Vocoder 


class Tacotron2(LightningModule):
    def __init__(self, encoder: LightningModule, decoder: LightningModule, 
                 postnet: LightningModule, mask_padding: bool=True,
                 n_mel_channels: int=80, n_frames_per_step: int=3,
                 symbols_lang: str="en", symbols_embedding_dim: int=512,
                 freeze_text: bool=False, load_pretrained_text: str=None,
                 multi_speaker: bool=False, spker_embedding_dim: int=64,
                 n_spker: int=1, speaker_loss_weight: float=0.02,
                 speaker_classifier: AdversarialClassifier=None,
                 vocoder: Vocoder=None):

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

        self.vocoder = vocoder
        if self.vocoder != None:
            for v in self.vocoder.generator.parameters():
                v.requires_grad = False


    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, spkerID = batch

        return ((text_padded, input_lengths, mel_padded, output_lengths, spkerID),
                (mel_padded, gate_padded, spkerID))

    def parse_output(self, outputs: Dict, output_lengths=None):
        if self.hparams.mask_padding and output_lengths is not None:
            output_total_length = outputs["mel"].size(2)
            mask = ~get_mask_from_lengths(output_lengths, output_total_length)
            mask = mask.expand(self.hparams.n_mel_channels, mask.size(0), mask.size(1))
            if mask.size(2)%self.hparams.n_frames_per_step != 0 :
                to_append = torch.ones( mask.size(0), mask.size(1), (self.hparams.n_frames_per_step-mask.size(2)%self.hparams.n_frames_per_step) ).bool()
                mask = torch.cat([mask, to_append], dim=-1)
            mask = mask.permute(1, 0, 2)

            outputs["mel_raw"].data.masked_fill_(mask, 0.0)
            outputs["mel"].data.masked_fill_(mask, 0.0)
            outputs["gate"].data.masked_fill_(mask[:, 0, :], 1e3)

        return outputs
    
    def compute_loss(self, model_output: Dict, targets: Dict):
        for t in targets.values():
            if t != None:
                t.requires_grad = False
        
        mel_target, gate_target = targets["mel"], targets["gate"].view(-1, 1)

        mel_loss = torch.nn.MSELoss()(model_output["mel_raw"], mel_target) + \
            torch.nn.MSELoss()(model_output["mel"], mel_target)

        gate_out = model_output["gate"].view(-1, 1)
        gate_loss = torch.nn.BCEWithLogitsLoss()(gate_out, gate_target)

        loss = {'mel_loss': mel_loss, 'gate_loss': gate_loss}

        if "speaker" in model_output:
            speaker_predict = model_output["speaker"].transpose(1, 2) # (B, T, n_speaker) -> (B, n_speaker, T)
            speaker_target = targets["speaker"].unsqueeze(1).repeat(1, speaker_predict.size(2)) # (B) -> (B, T)
            speaker_loss = torch.nn.CrossEntropyLoss()(speaker_predict, speaker_target)
            speaker_loss = speaker_loss * self.hparams.speaker_loss_weight
            loss['speaker_loss'] = speaker_loss

        loss['loss'] = sum(loss.values())
        return loss

    def forward(self, inputs: Dict):
        inputs["text_len"], inputs["mel_len"] = inputs["text_len"].data, inputs["mel_len"].data

        embedded_inputs = self.embedding(inputs["text"]).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, inputs["text_len"])

        outputs = dict({})
        if self.hparams.multi_speaker:
            if self.speaker_classifier != None:
                outputs["speaker"] = self.speaker_classifier(encoder_outputs)
            speaker_embedded = self.speaker_embedding(inputs["speaker"])[:, None]
            speaker_embedded = speaker_embedded.repeat(1, encoder_outputs.size(1), 1)
            encoder_outputs = torch.cat((encoder_outputs, speaker_embedded), dim=2)

        outputs["mel_raw"], outputs["gate"], outputs["align"] = self.decoder(
            encoder_outputs, inputs["mel"], memory_lengths=inputs["text_len"])

        postnet_outputs = self.postnet(outputs["mel_raw"])
        outputs["mel"] = outputs["mel_raw"] + postnet_outputs

        return self.parse_output(outputs, inputs["mel_len"])

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


    def mel2wav(self, mel) -> torch.Tensor:
        audio = self.vocoder.mel2wav(mel.T)
        return audio

    def log_loss(self, loss: Dict, prefix: str="training"):
        for k, v in loss.items():
            self.log(f"{prefix}.{k}", v)

    def log_validation(self, reduced_loss: Dict, y: Dict, y_pred: Dict, iteration: int):
        self.log_loss(reduced_loss, "validation")

        # plot distribution of parameters
        for tag, value in self.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.experiment.add_histogram(tag, value, iteration)

        # random select a sample for logging
        idx = random.randint(0, y_pred["align"].size(0) - 1)
        mel_len = y["mel_len"][idx].data

        # plot alignment, gate target and predicted
        self.logger.experiment.add_image(
            "alignment",
            plot_alignment_to_numpy(y_pred["align"][idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.logger.experiment.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                y["gate"][idx].data.cpu().numpy(),
                torch.sigmoid(y_pred["gate"][idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')

        # plot mel target and predicted
        mel_pred = y_pred["mel"][idx, :, :mel_len].data.cpu().numpy()
        mel_target = y["mel"][idx, :, :mel_len].data.cpu().numpy()
        self.logger.experiment.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_target),
            iteration, dataformats='HWC')
        self.logger.experiment.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_pred),
            iteration, dataformats='HWC')

        # synthesis audio from mel-spec
        audio_pred = self.mel2wav(mel_pred)
        audio_target = self.mel2wav(mel_target)
        self.logger.experiment.add_audio("audio_pred", audio_pred, iteration, self.vocoder.config.sampling_rate, )
        self.logger.experiment.add_audio("audio_target", audio_target, iteration, self.vocoder.config.sampling_rate)

    def training_step(self, batch: Dict, batch_idx):
        y_pred = self(batch)
        loss = self.compute_loss(y_pred, batch)
        self.log_loss(loss)
        return loss['loss']

    def validation_step(self, batch : Dict, batch_idx):
        y_pred = self(batch)
        loss = self.compute_loss(y_pred, batch)
        self.log_validation(loss, batch, y_pred, self.global_step)
        return loss['loss']