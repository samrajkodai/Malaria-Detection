from distutils.command.config import config
import os
import argparse

import yaml

def read_params(config_path):
    with open(config_path) as yaml_file:
        config=yaml.safe_load(yaml_file)

        return config

def get_data(config_path):
    config=read_params(config_path)
    train_path=config['load_data']['train_images']

    return train_path

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="config/params.yaml")
    parsed_args=args.parse_args()
    data=get_data(config_path=parsed_args.config)