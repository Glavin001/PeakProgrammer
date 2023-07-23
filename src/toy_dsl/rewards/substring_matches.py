from typing import List

def find_overlapping_tokens(encodings, char_range):
    # Tokenize the text
    # encodings = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=512)

    # Get the list of tokens and their corresponding start and end positions in the original text
    offsets = encodings.offset_mapping

    # Initialize the list to store the token indices
    token_indices = []

    start, end = char_range

    # Find the tokens that overlap with the given character range
    for token_idx, (token_start, token_end) in enumerate(offsets):
        # if start < token_end and end > token_start:  # Check for overlap
        if start < token_end and end > token_start:  # Check for overlap
            token_indices.append(token_idx)  # Token indices are 0-indexed

    return token_indices

def get_matching_ranges(text, search):
    all_match_indices = []
    start = 0
    while True:
        match_index = text.find(search, start)
        if match_index == -1:
            break
        all_match_indices.append((match_index, match_index + len(search)))
        start = match_index + 1
    return all_match_indices

def reward_substring_matches_single(text: str, encodings, searches: List[str], max_reward: float = 1.0, min_reward: float = 0.1) -> List[float]:
    """
    For each search string, find all matching ranges then find all overlapping tokens. Give max_reward for each of the overlapping tokens.
    Return a list of rewards for each token in the text. Non-overlapping tokens get 0 reward.
    """
    all_match_ranges = []
    for search in searches:
        all_match_ranges += get_matching_ranges(text, search)
    # all_token_indices = []
    all_token_ranges = []
    for match_range in all_match_ranges:
        # all_token_indices += find_overlapping_tokens(encodings, match_range)
        overlapping_range = find_overlapping_tokens(encodings, match_range)
        # all_token_indices += [overlapping_range[-1]]
        all_token_ranges.append(overlapping_range)
    rewards = [0.0] * len(encodings['input_ids'])

    # for token_index in all_token_indices:
    #     rewards[token_index] = max_reward

    # reward tokens
    for token_range in all_token_ranges:
        start = token_range[0]
        end = token_range[-1]
        # reward with min_reward for all tokens in the range
        # for token_index in range(start, end + 1):
        #     rewards[token_index] = min_reward

        # reward exponential decay for tokens before the range
        # for token_index in range(start - 1, -1, -1):
        #     rewards[token_index] = rewards[token_index + 1] * 0.9

        # reward exponentially from start token with min_reward to end token with max_reward
        for token_index in range(start, end + 1):
            if start == end:
                rewards[token_index] = max_reward
            else:
                rewards[token_index] = min_reward + (max_reward - min_reward) * (token_index - start) / (end - start)

        # reward with max_reward for the last token in the range
        rewards[end] = max_reward

    return rewards

def make_reward_substring_matches(searches: List[str], max_reward: float = 1.0, max_length: int = 512):
    def reward_substring_matches(samples: List[str], prompts: List[str], outputs: List[str], tokenizer, **kwargs) -> List[List[float]]:
        reward_list = []
        for sample_index, row in enumerate(zip(samples, prompts, outputs)):
            sample, prompt, output = row

            # parsed_sample = parse_sample(sample)

            encodings = tokenizer(output, return_offsets_mapping=True, truncation=True, max_length=max_length)

            rewards = reward_substring_matches_single(output, encodings, searches, max_reward)

            reward_list.append(rewards)

        return reward_list

    return reward_substring_matches
