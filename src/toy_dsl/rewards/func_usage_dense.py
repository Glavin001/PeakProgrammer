from typing import List
from utils import parse_sample

from .substring_matches import make_reward_substring_matches
from .func_usage import all_func_names, count_used_func_names

def reward_func_usage_dense(samples: List[str], prompts: List[str], outputs: List[str], tokenizer, **kwargs) -> List[List[float]]:
    """
    Reward each token based on the function names it uses
    Penalize using a function more than the count of expected occurrences
    """
    max_reward = 1
    max_penalty = -1
    reward_list = []
    for sample_index, row in enumerate(zip(samples, prompts, outputs)):
        sample, prompt, output = row

        parsed_sample = parse_sample(sample)
        generated_code = parsed_sample["code"]
        expected_code = kwargs['expected'][sample_index]

        expected_used = count_used_func_names(expected_code)
        generated_used = count_used_func_names(generated_code)

        expected_used_names: List[str] = list(set(expected_used.keys()))
        # map over expected_used_names and append "(" to each
        expected_used_names = [f"{name}(" for name in expected_used_names]

        max_length = 512 # FIXME
        rewards_fn = make_reward_substring_matches(expected_used_names, max_reward=max_reward, max_length=max_length)
        rewards = rewards_fn([sample], [prompt], [output], tokenizer, **kwargs)[0]

        reward_list.append(rewards)

    return reward_list


        