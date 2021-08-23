import os
import torch

from pytorch_lightning.utilities.cli import LightningCLI

from model.tacotron import Tacotron2
from utils.dataset import TextMelDataModule
from utils.trainer import MyTestTubeTrainer

    
class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)

        parser.link_arguments("data.n_mel_channels", "model.n_mel_channels")
        parser.link_arguments("data.n_mel_channels", "model.decoder.init_args.n_mel_channels")
        parser.link_arguments("data.n_mel_channels", "model.postnet.init_args.n_mel_channels")
        parser.link_arguments("data.symbols_lang", "model.symbols_lang")
        parser.link_arguments("model.n_frames_per_step", "data.n_frames_per_step")
        parser.link_arguments("model.n_frames_per_step", "model.decoder.init_args.n_frames_per_step")
        parser.link_arguments("model.encoder.encoder_embedding_dim", "model.decoder.init_args.encoder_embedding_dim")

    def before_instantiate_classes(self) -> None:
        pass

    def before_fit(self):
        print("Now fitting")

    def after_fit(self):
        pass


if __name__ == "__main__":
    cli = MyLightningCLI(Tacotron2, TextMelDataModule,
                         trainer_class=MyTestTubeTrainer,
                         save_config_overwrite=True)
