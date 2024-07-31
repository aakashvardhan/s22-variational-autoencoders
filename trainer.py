import os
from argparse import ArgumentParser

import lightning as L
import torch
from pl_bolts.datamodules import CIFAR10DataModule

from datamodule import MNISTDataModule
from lit_vae import VAE
from utils import generate_wrong_label_images, plt_result


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="mnist", help="The dataset to use."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="The batch size to use."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="The number of epochs to train for.",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Whether to run the model in debug mode.")

    return vars(parser.parse_args())


L.seed_everything(1234, workers=True)
torch.cuda.empty_cache()

args = parse_args()

if args["dataset"] == "mnist":
    data_module = MNISTDataModule(".", batch_size=args["batch_size"])
elif args["dataset"] == "cifar10":
    data_module = CIFAR10DataModule(".", batch_size=args["batch_size"])


data_module.prepare_data()
data_module.setup()
model = VAE()

if args["debug"]:
    trainer = L.Trainer(fast_dev_run=True)
else:
    trainer = L.Trainer(max_epochs=args["epochs"])
trainer.fit(model, data_module)

data_loader = data_module.val_dataloader()

result = generate_wrong_label_images(model, data_loader)

plt_result(label=args["dataset"], result=result)
