from typing import List
import numpy as np
from utils import clamp_reward, parse_sample

from lang import Interpreter

interpreter = Interpreter()

# def make_reward_output_length_diff(max_possible_length=10):
def make_reward_output_length_diff():
    def reward_length_diff(samples: List[str], prompts: List[str], outputs: List[str], tokenizer, **kwargs) -> List[List[float]]:
        reward_list = []
        for sample, prompt, output in zip(samples, prompts, outputs):
            if len(output) == 0:
                reward_list.append([])
                continue

            tokens = tokenizer.tokenize(output, add_special_tokens=False)

            parsed_sample = parse_sample(sample)
            code = parsed_sample["code"]

            expected_output = parsed_sample["output"]
            observed_output = interpreter(code)

            expected_length = len(expected_output)
            observed_length = len(observed_output)
            max_possible_length = max(1, expected_length, observed_length)

            last_reward = reward_length_diff_nums(expected_length, observed_length, max_possible_length=max_possible_length)

            reward = [0.0] * len(tokens)
            reward[-1] = last_reward

            reward_list.append(reward)

        return reward_list

    return reward_length_diff


def reward_length_diff_nums(expected_length, observed_length, max_possible_length=10):
    length_diff = abs(expected_length - observed_length)
    steepness = 2
    reward = np.exp(-steepness * length_diff / max_possible_length)
    final_reward = clamp_reward(reward)
    return final_reward
