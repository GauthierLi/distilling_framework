import argparse
import importlib.util

from typing import List
from distill.loger import gaulog

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--configs", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--workdir", type=str, required=False, help="Path to sae the output"
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="distill",
        choices=["distill"],
        help="Path to sae the output",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to the checkpoint file"
    )
    return parser.parse_args()


def parse_config(config_path: str):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def assert_all(components: List[str]):
    assert "dataloader" in components, "dataloader is missing in the config file"
    assert "models" in components, "models is missing in the config file"
    assert "loss" in components, "loss is missing in the config file"
    assert (
        "distill_structure" in components
    ), "distill_structure is missing in the config file"
    assert "optimizer" in components, "optimizer is missing in the config file"
    assert "trainer" in components, "trainer is missing in the config file"
    assert "scheduler" in components, "scheduler is missing in the config file"


def get_structure(config):
    models = config.models
    distill_structure = config.distill_structure
    structure = distill_structure(
        teacher=models["teacher"],
        student=models["student"],
        optimizer=config.optimizer,
        loss=config.loss,
    )
    return structure


def get_trainer(config):
    # distill_structure = get_structure(config)
    device = config.trainer["device"]
    epochs = config.trainer["epochs"]
    models = config.models
    optimizer = config.optimizer
    loss = config.loss
    scheduler = config.scheduler  # can be None
    dataloader = config.dataloader
    trainer = config.trainer["type"](
        device=device,
        epochs=epochs,
        models=models,
        optimizer=optimizer,
        loss=loss,
        structure=config.distill_structure,
        scheduler=scheduler,
        dataloader=dataloader,
        validate_period=config.trainer["validate_period"],
        save_dir=config.trainer["save_dir"],
    )
    return trainer


def main(args: argparse.Namespace):
    config = parse_config(args.configs)
    all_components = dir(config)
    assert_all(all_components)
    trainer = get_trainer(config)
    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    main(args)
