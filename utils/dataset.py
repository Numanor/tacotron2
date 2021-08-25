import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from text import text_to_sequence


class TextMelDataModule(LightningDataModule):
    def __init__(self, meta_train: str, meta_valid: str, mel_dir: str,
                 n_mel_channels: int=80, n_frames_per_step: int=3,
                 batch_size: int=32, symbols_lang: str="en",
                 text_cleaners: List[str]=["basic_cleaners"],
                 multi_speaker: bool=False, speaker_item_idx: int=None):

        super(TextMelDataModule, self).__init__()
        
        self.meta_train = meta_train
        self.meta_valid = meta_valid
        self.mel_dir = Path(mel_dir)
        self.n_mel_channels = n_mel_channels
        self.batch_size = batch_size
        self.symbols_lang = symbols_lang
        self.text_cleaners = text_cleaners
        self.collate_fn = TextMelCollate(n_frames_per_step)
        self.speakerItemIdx = speaker_item_idx if multi_speaker else None
    
    def setup(self, stage: Optional[str]):
        self.trainset = TextMelDataset(self.meta_train, self.mel_dir, self.text_cleaners,
                                       self.symbols_lang, self.n_mel_channels, self.speakerItemIdx)
        self.validset = TextMelDataset(self.meta_valid, self.mel_dir, self.text_cleaners,
                                       self.symbols_lang, self.n_mel_channels, self.speakerItemIdx)
    
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size,
                          shuffle=True, num_workers=30, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size,        
                          shuffle=False, num_workers=56, collate_fn=self.collate_fn)


class TextMelDataset(torch.utils.data.Dataset):
    """
        1) loads filepath,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) loads mel-spectrograms from mel files
    """
    def __init__(self, fname: str, mel_dir: Path, 
                 text_cleaners: List[str], symbols_lang: str,
                 n_mel_channels: int,
                 speakerItemIdx: int=None):
        self.text_cleaners  = text_cleaners
        self.mel_dir = mel_dir
        self.symbols_lang   = symbols_lang
        self.n_mel_channels = n_mel_channels
        self.speakerItemIdx = speakerItemIdx
        self.f_list = self.files_to_list(fname)
        random.shuffle(self.f_list)

    def files_to_list(self, file_path):
        f_list = []
        with open(file_path, encoding = 'utf-8') as f:
            for line in f:
                parts = line.strip().strip('\ufeff').split('|') #remove BOM
                sample = [parts[-1], parts[0]] # [text, mel_file_path]
                if self.speakerItemIdx != None:
                    sample.append(int(parts[self.speakerItemIdx])-1)
                f_list.append(sample)
        return f_list

    def get_mel_text_pair(self, text, file_path, spkerID=None):
        text = self.get_text(text)
        mel = self.get_mel(file_path)
        if spkerID != None:
            return (text, mel, spkerID)
        else:
            return (text, mel)

    def get_mel(self, file_path):
        #stored melspec: np.ndarray [shape=(T_out, num_mels)]
        #in Pytorch [shape=(num_mels, T_out)]
        melspec = torch.from_numpy(np.load(self.mel_dir/f"{file_path}.npy").T)
        assert melspec.size(0) == self.n_mel_channels, (
            'Mel dimension mismatch: given {}, expected {}'.format(melspec.size(0), self.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners, self.symbols_lang))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(*self.f_list[index])

    def __len__(self):
        return len(self.f_list)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized, speaker_ID(optional)]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        output = {
            "text": text_padded,
            "text_len": input_lengths,
            "mel": mel_padded,
            "gate": gate_padded,
            "mel_len": output_lengths
        }

        if len(batch[0]) > 2:
            spkerID = torch.LongTensor(len(ids_sorted_decreasing))
            for i in range(len(ids_sorted_decreasing)):
                spkerID[i] = batch[ids_sorted_decreasing[i]][2]
            output["speaker"] = spkerID
        
        return output
