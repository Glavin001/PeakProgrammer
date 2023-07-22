from typing import List
import numpy as np
from utils import clamp_reward, parse_sample
from lang import Interpreter, list_manip_dsl_gen

all_func_names = list(list_manip_dsl_gen.keys())

def count_used_func_names(code: str) -> dict:
    # count how many times each function name is used in the code
    # return a dict with function names as keys and counts as values
    used_func_names = {}
    for func_name in all_func_names:
        used_func_names[func_name] = code.count(func_name + "(")
    return used_func_names

def total_func_count(used_func_names: dict):
    return sum(used_func_names.values())

# def diff_func_counts(used_a: dict, used_b: dict):
#     # return the difference in counts of function names used
#     diff = 0
#     for func_name in all_func_names:
#         diff += abs(used_a[func_name] - used_b[func_name])
#     return diff

def missing_func_count(used_a: dict, used_b: dict):
    # return the difference in counts of function names used
    diff = 0
    for func_name in all_func_names:
        diff += max(0, used_a[func_name] - used_b[func_name])
    return diff

# def reward_func_usage_text(expected: str, generated: str) -> float:
#     # maximum value is 1
#     # minimum value is 0 when count of different function names used is >= # of function names in expected
#     # return 1 - (count of different function names used / # of function names in expected)
#     expected_used = count_used_func_names(expected)
#     generated_used = count_used_func_names(generated)
#     diff = diff_func_counts(expected_used, generated_used) / 2
#     max_diff = total_func_count(expected_used)
#     return 1 - (diff / max_diff)

# def reward_output_length(expected_length, observed_length, max_possible_length=10):
#     length_diff = abs(expected_length - observed_length)
#     steepness = 2
#     reward = np.exp(-steepness * length_diff / max_possible_length)
#     final_reward = clamp_reward(reward)
#     return final_reward

def reward_func_usage_nums(perc_used: float, perc_missing: float) -> float:
    used_steepness = 1
    used_reward = np.exp(used_steepness*(perc_used - 1))
    return clamp_reward(used_reward)

def reward_func_usage_nums_text(expected: str, generated: str) -> float:
    # maximum value is 1
    # minimum value is 0 when count of different function names used is >= # of function names in expected
    # return 1 - (count of different function names used / # of function names in expected)
    expected_used = count_used_func_names(expected)
    generated_used = count_used_func_names(generated)
    diff = missing_func_count(expected_used, generated_used)
    max_diff = total_func_count(expected_used)
    if max_diff == 0:
        return 0
    perc_missing = diff / max_diff
    perc_used = 1 - (diff / max_diff)
    # print("perc_used:", perc_used)
    # print("perc_missing:", perc_missing)
    # return reward_func_usage_nums(perc_used, perc_missing)
    return reward_func_usage_nums(perc_used, 0.1 if perc_missing > 0 else 0)

def reward_func_usage(samples: List[str], prompts: List[str], outputs: List[str], tokenizer, **kwargs) -> List[List[float]]:
    reward_list = []
    # for sample, prompt, output in zip(samples, prompts, outputs):
    # add an index
    # for sample_index, sample in enumerate(samples):
    # for sample_index, sample, prompt, output in enumerate(zip(samples, prompts, outputs)):
    # for sample_index, [sample, prompt, output] in enumerate(zip(samples, prompts, outputs)):
    for sample_index, row in enumerate(zip(samples, prompts, outputs)):
        sample, prompt, output = row

        parsed_sample = parse_sample(sample)
        generated_code = parsed_sample["code"]
        expected_code = kwargs['expected'][sample_index]

        reward = reward_func_usage_nums_text(expected_code, generated_code)

        reward_list.append(reward)

    return reward_list
