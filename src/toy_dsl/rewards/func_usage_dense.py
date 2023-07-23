from typing import List
from utils import parse_sample

# from .substring_matches import make_reward_substring_matches
from .substring_matches import get_matching_ranges, find_overlapping_tokens
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


def reward_substring_matches_single(text: str, encodings, search: str, max_count: int = 0, max_reward: float = 1.0, min_reward: float = 0.1) -> List[float]:
    """
    For each search string, find all matching ranges then find all overlapping tokens. Give max_reward for each of the overlapping tokens.
    Return a list of rewards for each token in the text. Non-overlapping tokens get 0 reward.
    """

    all_match_ranges = []
    all_match_ranges += get_matching_ranges(text, search)

    all_token_ranges = []
    for match_range in all_match_ranges:
        overlapping_range = find_overlapping_tokens(encodings, match_range)
        all_token_ranges.append(overlapping_range)

    rewards = [0.0] * len(encodings['input_ids'])

    for range_index, token_range in enumerate(all_token_ranges):
        start = token_range[0]
        end = token_range[-1]

        # w = 1.0 if range_index < max_count else -1.0
        # w = max_reward if range_index < max_count else -min_reward
        w = max_reward if range_index < max_count else 0.0

        for token_index in range(start, end + 1):
            if start == end:
                rewards[token_index] = w * max_reward
            else:
                rewards[token_index] = w * (min_reward + (max_reward - min_reward) * (token_index - start) / (end - start))

        rewards[end] = w * max_reward

    return rewards


def make_reward_substring_matches(searches: List[str], max_reward: float = 1.0, max_length: int = 512):
    def reward_substring_matches(samples: List[str], prompts: List[str], outputs: List[str], tokenizer, **kwargs) -> List[List[float]]:
        reward_list = []
        for sample_index, row in enumerate(zip(samples, prompts, outputs)):
            sample, prompt, output = row

            encodings = tokenizer(output, return_offsets_mapping=True, truncation=True, max_length=max_length)

            rewards = [0.0] * len(encodings['input_ids'])
            for search in searches:
                match_count = kwargs['expected'][sample_index].count(search)
                rewards_for_search = reward_substring_matches_single(output, encodings, search, match_count, max_reward)
                # rewards = [max(reward, new_reward) for reward, new_reward in zip(rewards, rewards_for_search)]
                rewards = [(reward + new_reward) for reward, new_reward in zip(rewards, rewards_for_search)]

            reward_list.append(rewards)

        return reward_list

    return reward_substring_matches
