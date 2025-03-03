import argparse
from distill.loger import gaulog 

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--configs", type=str, required=True, help="Path to the config file")
    parser.add_argument("--workdir", type=str, required=True, help="Path to sae the output")
    parser.add_argument("--task-name", type=str, default="distill",choices=["distill"], help="Path to sae the output")
    return parser.parse_args()
