from typing import List
from utils import parse_sample

# from .substring_matches import make_reward_substring_matches
from .substring_matches import get_matching_ranges, find_overlapping_tokens
from .func_usage import all_func_names, count_used_func_names

from lang import Interpreter

interpreter = Interpreter()

MESSAGES = [
    'invalid syntax'
]

EOS_TOKEN = "<|endoftext|>"

def penalize_syntax_errors(samples: List[str], prompts: List[str], outputs: List[str], tokenizer, **kwargs) -> List[List[float]]:
    max_penalty = -1
    max_length = 512 # FIXME
    reward_list = []
    for sample_index, row in enumerate(zip(samples, prompts, outputs)):
        sample, prompt, output = row

        # code = sample.split("Function:")[1].strip()
        code = output

        if code.endswith(EOS_TOKEN):
            code = code[:-len(EOS_TOKEN)]

        tokens = tokenizer.tokenize(output, add_special_tokens=False)
        rewards = [0.0] * len(tokens)

        try:
            interpreted_output = interpreter.eval(code)
        except Exception as e:
            # print(e)
            should_penalize = any([msg in str(e) for msg in MESSAGES])

            if not should_penalize:
                reward_list.append(rewards)
                continue

            # print(e.offset)
            # print(code[e.offset - 1])
            # print(code[e.end_offset - 1])

            if not hasattr(e, 'offset'):
                reward_list.append(rewards)
                continue

            if hasattr(e, 'end_offset'):
                match_range = (e.offset - 1, e.end_offset - 1)
            else:
                print(f"no end_offset for: {e}")
                match_range = (e.offset - 1, e.offset)

            encodings = tokenizer(output, return_offsets_mapping=True, truncation=True, max_length=max_length)

            token_range = find_overlapping_tokens(encodings, match_range)

            if len(token_range) == 0:
                print(f"no overlapping tokens for {match_range} in {output}")
                reward_list.append(rewards)
                continue

            start = token_range[0]
            end = token_range[-1]

            # w = 1.0 if range_index < max_count else -1.0
            # w = max_reward if range_index < max_count else -min_reward
            w = -1.0
            max_reward = 1.0
            min_reward = 0.1

            for token_index in range(start, end + 1):
                if start == end:
                    rewards[token_index] = w * max_reward
                else:
                    rewards[token_index] = w * (min_reward + (max_reward - min_reward) * (token_index - start) / (end - start))

            rewards[end] = w * max_reward

        reward_list.append(rewards)

    return reward_list


# def reward_o(text: str, encodings, search: str, max_count: int = 0, max_reward: float = 1.0, min_reward: float = 0.1) -> List[float]:
#     """
#     For each search string, find all matching ranges then find all overlapping tokens. Give max_reward for each of the overlapping tokens.
#     Return a list of rewards for each token in the text. Non-overlapping tokens get 0 reward.
#     """

#     all_match_ranges = []
#     all_match_ranges += get_matching_ranges(text, search)

#     all_token_ranges = []
#     for match_range in all_match_ranges:
#         overlapping_range = find_overlapping_tokens(encodings, match_range)
#         all_token_ranges.append(overlapping_range)

#     rewards = [0.0] * len(encodings['input_ids'])

#     for range_index, token_range in enumerate(all_token_ranges):
#         start = token_range[0]
#         end = token_range[-1]

#         # w = 1.0 if range_index < max_count else -1.0
#         w = max_reward if range_index < max_count else -min_reward

#         for token_index in range(start, end + 1):
#             if start == end:
#                 rewards[token_index] = w * max_reward
#             else:
#                 rewards[token_index] = w * (min_reward + (max_reward - min_reward) * (token_index - start) / (end - start))

#         rewards[end] = w * max_reward

#     return rewards
