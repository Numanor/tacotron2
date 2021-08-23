
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.test_tube import TestTubeLogger

class MyTestTubeTrainer(Trainer):
    @property
    def log_dir(self):
        if isinstance(self.logger, TestTubeLogger):
            exp = self.logger.experiment
            dirpath = exp.get_data_path(exp.name, exp.version)
            dirpath = self.accelerator.broadcast(dirpath)
            return dirpath
        else:
            return super().log_dir()