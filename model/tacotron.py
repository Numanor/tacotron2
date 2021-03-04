from math import sqrt

import torch

from .model import Encoder, Decoder, Postnet, AdversarialClassifier
from utils.utils import to_gpu, get_mask_from_lengths


class Tacotron2Loss(torch.nn.Module):
    def __init__(self, hparams):
        super(Tacotron2Loss, self).__init__()
        self.speaker_loss_weight = hparams.speaker_loss_weight

    def forward(self, model_output, targets):
        mel_target, gate_target, speaker_target = targets
        speaker_target.requires_grad = False
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _, speaker_out = model_output
        gate_out = gate_out.view(-1, 1)

        mel_loss = torch.nn.MSELoss()(mel_out, mel_target) + \
            torch.nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = torch.nn.BCEWithLogitsLoss()(gate_out, gate_target)

        # Compute speaker adversarial training loss
        #
        # The speaker adversarial loss should be computed against each element of the encoder output.
        #
        # In Google's paper (https://arxiv.org/abs/1907.04448), it is mentioned that:
        # 'We impose this adversarial loss separately on EACH ELEMENT of the encoded text sequence,...'
        #
        speaker_target = speaker_target.unsqueeze(1).repeat(1, speaker_out.size(2)) # [B] -> [B, T_in]
        speaker_loss = torch.nn.CrossEntropyLoss()(speaker_out, speaker_target) # speaker_out: [B, n_speakers, T_in]

        return mel_loss + gate_loss + speaker_loss * self.speaker_loss_weight


class Tacotron2(torch.nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = torch.nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.speaker_embedding = torch.nn.Embedding(
            hparams.n_speakers, hparams.speaker_embedding_dim)
        std = sqrt(2.0 / (hparams.n_speakers + hparams.speaker_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.speaker_embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.speaker_classifier = AdversarialClassifier(hparams.encoder_embedding_dim, [hparams.speaker_hidden_dim, hparams.n_speakers])

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, speakers = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        speakers = to_gpu(speakers).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths, speakers),
            (mel_padded, gate_padded, speakers))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            if mask.size(2)%self.n_frames_per_step != 0 :
                to_append = torch.ones( mask.size(0), mask.size(1), (self.n_frames_per_step-mask.size(2)%self.n_frames_per_step) ).bool().to(mask.device)
                mask = torch.cat([mask, to_append], dim=-1)
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths, speakers = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        speaker_embeddings = self.speaker_embedding(speakers)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        speaker_outputs = self.speaker_classifier(encoder_outputs).transpose(1, 2)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, speaker_embeddings, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, speaker_outputs],
            output_lengths)

        return outputs

    def inference(self, inputs, speakers):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)

        speaker_embeddings = self.speaker_embedding(speakers)

        encoder_outputs = self.encoder.inference(embedded_inputs)

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs, speaker_embeddings)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
