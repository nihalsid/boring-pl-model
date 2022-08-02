import os
import random

import pytorch_lightning
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.profiler import SimpleProfiler

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from torchvision.models import resnet152


tmpdir = os.getcwd()

num_samples = 8192
max_on_gpu_samples = 64
batch_size = 2
mode = "GPUOnly"
# mode = "CPU&GPU"

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False


class DummyDataset(torch.utils.data.Dataset):

    def __len__(self):
        return num_samples

    def __getitem__(self, idx):
        return []


class RandomImageDataset(torch.utils.data.Dataset):

    def __init__(self, h, w, length):
        self.len = length
        self.data = torch.randn(length, 3, h, w)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


train = RandomImageDataset(224, 224, num_samples) if mode == "CPU&GPU" else DummyDataset()
train = DataLoader(train, batch_size=batch_size, num_workers=8)


class BoringModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.model = resnet152(pretrained=False)
        # self.model = SimpleMLP(in_channels=32, num_mlp_layers=150)
        self.register_buffer("batch", torch.randn([max_on_gpu_samples, batch_size, 3, 224, 224]))

    def forward(self, x):
        x_in = x if mode == "CPU&GPU" else self.batch[random.randint(0, max_on_gpu_samples - 1)]
        return self.model(x_in)

    @staticmethod
    def loss(batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, outputs) -> None:
        torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def validation_epoch_end(self, outputs) -> None:
        torch.stack([x['x'] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        output = self.model(batch)
        loss = self.loss(batch, output)
        self.log('fake_test_acc', loss)
        return {"y": loss}

    def test_epoch_end(self, outputs) -> None:
        torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


def run():
    # init model
    model = BoringModel()
    lr_monitor = LearningRateMonitor(logging_interval='step')

    profiler = SimpleProfiler(filename=f"run.pprof")
    logger = pytorch_lightning.loggers.TensorBoardLogger(save_dir="logs/")
    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=1,
        progress_bar_refresh_rate=1,
        precision=32,
        callbacks=[lr_monitor],
        profiler=profiler,
        logger=logger
    )

    # Train the model ⚡
    trainer.fit(model, train)


if __name__ == "__main__":
    run()
