# IMPORTS
import os
from os.path import basename
from pathlib import Path

import argparse
from sconf import Config
import datetime

import comet_ml
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger

# Utilities
import pytorch_lightning as pl
import torch

# import lightning & further tools
import lightning_fabric
from lightning_module import DonutModelPLModule, DonutDataPLModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint


from donut import DonutDataset


class ProgressBar(pl.callbacks.TQDMProgressBar):
    def __init__(self, config):
        super().__init__()
        self.enable = True
        self.config = config

    def disable(self):
        self.enable = False

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        items["exp_name"] = f"{self.config.get('exp_name', '')}"
        items["exp_version"] = f"{self.config.get('exp_version', '')}"
        return items


def train(config):
    # set lightning seed (for pytorch-lightning version >= 2.0.0)
    lightning_fabric.utilities.seed.seed_everything(42, workers=True)

    model_module = DonutModelPLModule(config)
    data_module = DonutDataPLModule(config)

    # add datasets to the data module
    datasets = {"train": [], "validation": []}

    for i, dataset_name_or_path in enumerate(config.dataset_name_or_paths):
        task_name = os.path.basename(dataset_name_or_path)

        for split in ["train", "validation"]:
            datasets[split].append(
                DonutDataset(
                    dataset_name_or_path=dataset_name_or_path,
                    donut_model=model_module.model,
                    max_length=config.max_length,
                    split=split,
                    task_start_token=config.task_start_tokens[i]
                    if config.get("task_start_tokens", None)
                    else f"s_{task_name}>",
                    prompt_end_token="<s_answer>"
                    if "docvqa" in dataset_name_or_path
                    else f"<s_{task_name}>",
                    sort_json_key=config.sort_json_key,
                )
            )

    data_module.train_datasets = datasets["train"]
    data_module.val_datasets = datasets["validation"]

    # COMET LOGGER
    comet_logger = CometLogger(
        api_key=os.environ["COMET_API_KEY"],
        project_name=os.environ["COMET_PROJECT_NAME"],
    )
    comet_logger.log_hyperparams(config)

    # COMET EXPERIMENT
    comet_experiment = Experiment(
        api_key=os.environ["COMET_API_KEY"],
        project_name=os.environ["COMET_PROJECT_NAME"],
    )
    comet_experiment.log_parameters(config)

    # VISUALISATION
    lr_callback = LearningRateMonitor(logging_interval="step")
    bar = ProgressBar(config)

    trainer = pl.Trainer(
        logger=comet_logger,
        strategy="auto",
        accelerator=os.environ["DONUT_ACCELERATOR"],
        precision="16-mixed",
        callbacks=[lr_callback, bar],
    )

    history = trainer.fit(model_module, data_module)

    model_path = "../modelfile.pth"
    torch.save(model_module.model.state_dict(), model_path)
    comet_experiment.log_model("donut-model", model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=True)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)

    config.exp_name = basename(args.config).split(".")[0]
    config.exp_version = (
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not args.exp_version
        else args.exp_version
    )

    train(config)
