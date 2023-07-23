from typing import List
import numpy as np
from utils import clamp_reward, parse_sample

from lang import Interpreter

interpreter = Interpreter()

def make_reward_closing_brackets(max_reward: float = 1.0):
    def reward_closing_brackets(samples: List[str], prompts: List[str], outputs: List[str], tokenizer, **kwargs) -> List[List[float]]:
        reward_list = []
        for sample, prompt, output in zip(samples, prompts, outputs):
            if len(output) == 0:
                reward_list.append([])
                continue

            tokens = tokenizer.tokenize(output, add_special_tokens=False)

            reward = reward_closing_brackets_single(tokens, max_reward=max_reward)

            reward_list.append(reward)

        return reward_list

    return reward_closing_brackets


def reward_closing_brackets_single(tokens: List, max_reward: float = 1.0):
    """
    Reward closing brackets
    For every ")" so long as there are more "(" before it, give a reward
    """
    num_open = 0
    rewards_per_token: List[float] = [0.0] * len(tokens)
    for token_index, token in enumerate(tokens):
        # "(" and ")" may be part of a token, not the entire token
        # so we need to count the number of "(" and ")" in each token
        num_open += token.count("(")
        count_end_brackets = token.count(")")
        # count extra ")"s
        # if num_open = 1 and count_end_brackets = 2 then count_extra_end_brackets = 1
        # if num_open = 2 and count_end_brackets = 2 then count_extra_end_brackets = 0
        # if num_open = 0 and count_end_brackets = 1 then count_extra_end_brackets = 1
        # count_extra_end_brackets = max(0, count_end_brackets - num_open)
        count_extra_end_brackets = max(0, count_end_brackets - max(0, num_open))

        num_open -= count_end_brackets
        if count_end_brackets > 0:
            if num_open >= 0:
                rewards_per_token[token_index] = max_reward
            else:
                # rewards_per_token[token_index] = -max_reward
                # proportate to the # of extra ")"s
                # rewards_per_token[token_index] = -max_reward * -num_open
                rewards_per_token[token_index] = -max_reward * count_extra_end_brackets

    # penalize last token if there are extra "("s
    if num_open > 0:
        rewards_per_token[-1] = -max_reward * num_open

    return rewards_per_token
