import torch
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.utilities.cli import LightningCLI

from model.tacotron import Tacotron2

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--cudnn.enabled", default=True)
        parser.add_argument("--cudnn.benchmark", default=False)
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)

        parser.link_arguments("model.n_mel_channels", "model.decoder.init_args.n_mel_channels")
        parser.link_arguments("model.n_frames_per_step", "model.decoder.init_args.n_frames_per_step")
        parser.link_arguments("model.encoder.encoder_embedding_dim", "model.decoder.init_args.encoder_embedding_dim")
        parser.link_arguments("model.n_mel_channels", "model.postnet.init_args.n_mel_channels")

    def before_instantiate_classes(self) -> None:
        torch.backends.cudnn.enabled = self.config['cudnn']['enabled']
        torch.backends.cudnn.benchmark = self.config['cudnn']['benchmark']
        print("torch.backends.cudnn.enabled: ", torch.backends.cudnn.enabled)
        print("torch.backends.cudnn.benchmark: ", torch.backends.cudnn.benchmark)
        print(self.config)

    def before_fit(self):
        pass

    def after_fit(self):
        pass


if __name__ == "__main__":
    # cli = MyLightningCLI(Tacotron2, run=False)
    # cli = MyLightningCLI(Tacotron2, save_config_filename="out.yaml")
    cli = MyLightningCLI(Tacotron2, save_config_filename="out.yaml")