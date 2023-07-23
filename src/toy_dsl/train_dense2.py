import json
import logging
import pathlib
from typing import List
import numpy as np

import yaml
from lang import Interpreter, list_manip_dsl_gen

import trlx
from trlx.data.configs import TRLConfig

from rewards import reward_basic, reward_basic_dense, make_reward_token_cost, make_reward_output_length_diff, reward_func_usage_dense, make_reward_closing_brackets, make_reward_weighted_combo, penalize_syntax_errors

logger = logging.getLogger(__name__)

all_func_names = list(list_manip_dsl_gen.keys())

class DSLDataset:
    def __init__(self):
        with open("dataset/train.json", "r") as f:
            self.train_data = json.load(f)
        with open("dataset/test.json", "r") as f:
            self.test_data = json.load(f)
        logger.info("Sucessfully loaded the dataset")

    def load_datapoints(self, split="train"):
        if split == "train":
            for datapoint in self.train_data:
                if "ERROR" not in datapoint["input"]:
                    yield {
                        "prompt": datapoint["input"],
                        "original_output": datapoint["output"],
                        "expected": datapoint["output"],
                    }
        elif split == "test":
            for datapoint in self.test_data:
                yield {
                    "prompt": datapoint["input"],
                    "original_output": datapoint["output"],
                    "expected": datapoint["output"],
                }

interpreter = Interpreter()

EOS_TOKEN = "<|endoftext|>"

config_path = pathlib.Path(__file__).parent.joinpath("configs/trlx_ppo_config.yml")
with config_path.open() as f:
    default_config = yaml.safe_load(f)

MAX_LENGTH = 200
dense_reward_fn = make_reward_weighted_combo([
    reward_basic_dense,
    # make_reward_token_cost(MAX_LENGTH, 0.5, 0.01),
    make_reward_token_cost(MAX_LENGTH, 0.5, 0.01),
    reward_func_usage_dense,
    make_reward_closing_brackets(max_reward=1.0),
    make_reward_output_length_diff(),
    penalize_syntax_errors,
], [
    # 1.0,
    # 0.01,
    # 0.01,
    # 0.05,
    # 0.01,
    # 0.2,

    1.0,
    0.01,
    0.05,
    0.05,
    0.01,
    0.5,
])

USE_DENSE = True
reward_fn = dense_reward_fn if USE_DENSE else reward_basic

def main(hparams={}):
    final_config = default_config
    if USE_DENSE:
        final_config = default_config.copy()
        final_config["train"]["project_name"] = final_config["train"]["project_name"] + "_dense"
    config = TRLConfig.update(final_config, hparams)

    # Dataset
    dataset = DSLDataset()
    train_prompts = list(dataset.load_datapoints(split="train"))[:10000]
    test_prompts = list(dataset.load_datapoints(split="test"))[:100]

    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=test_prompts,
        config=config,
    )
    trainer.save_pretrained("dataset/trained_model")


if __name__ == "__main__":
    # TEST REWARD FUNTION
    # assert (reward_fn(["Input: 1 Output: [-4,-5,-2] Function: div_n(reverse([-2, -5, -4]),1)"])) == [1]
    # assert (reward_fn(["Input: 1 Output: [-4,-5,-2] Function: div_n(reverse([-2, -5, -a]),1)"])) == [-1]
    # assert (reward_fn(["Input: 1 Output: [-4,-5,-2] Function: div_n(reverse([-2, -5, -3]),1)"])) == [-0.5]

    main()
