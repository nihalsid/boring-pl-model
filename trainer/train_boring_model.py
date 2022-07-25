import os
import torch
from pytorch_lightning import LightningModule

from integrations.lightning.boring_model import RandomDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
# some other options for random data
from pytorch_lightning.callbacks import LearningRateMonitor

from model.mlp import SimpleMLP

tmpdir = os.getcwd()

num_samples = 100000
train = RandomDataset(32, num_samples)
train = DataLoader(train, batch_size=32)
val = RandomDataset(32, num_samples)
val = DataLoader(val, batch_size=32)
test = RandomDataset(32, num_samples)
test = DataLoader(test, batch_size=32)


class BoringModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.model = SimpleMLP(in_channels=32)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def loss(batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, outputs) -> None:
        torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        output = self.model(batch)
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

    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=10,
        progress_bar_refresh_rate=20,
        precision=32,
        callbacks=[lr_monitor]
    )

    # Train the model ⚡
    trainer.fit(model, train)


if __name__ == "__main__":
    run()
