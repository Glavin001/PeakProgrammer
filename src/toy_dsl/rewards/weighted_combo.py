from typing import List, Callable

# def calculate_weighted_reward(rewards: List[float], weights: List[float]) -> List[float]:
def calculate_weighted_reward(rewards: List[List[float]], weights: List[float]) -> List[float]:
    # Ensure the lengths of rewards and weights are the same
    if len(rewards) != len(weights):
        raise ValueError("Length of rewards and weights must be equal")

    # Ensure the sum of weights equals 1
    # if not 0.99 <= sum(weights) <= 1.01:
    #     raise ValueError("Sum of weights must be approximately 1")

    # Calculate the weighted reward
    # weighted_reward = sum(reward * weight for reward, weight in zip(rewards, weights))
    weighted_reward = [sum(reward * weight for reward, weight in zip(reward_list, weights)) for reward_list in zip(*rewards)]

    return weighted_reward

def make_reward_weighted_combo(
    reward_funcs: List[Callable],
    weights: List[float],
):
    def reward_weighted_combo(samples: List[str], prompts: List[str], outputs: List[str], tokenizer, **kwargs) -> List[List[float]]:

        # try:
            # map over reward_funcs and call each with samples, prompts, outputs, tokenizer, **kwargs
            # rewards_per_sample = [reward_func(samples, prompts, outputs, tokenizer, **kwargs) for reward_func in reward_funcs]
            rewards_per_func: List[List[List[float]]] = [reward_func(samples, prompts, outputs, tokenizer, **kwargs) for reward_func in reward_funcs]

            rewards_per_sample: List[List[List[float]]] = []
            for sample_index in range(len(samples)):
                rewards_per_func_per_sample: List[List[float]] = []
                for reward_func_index in range(len(reward_funcs)):
                    rewards_per_func_per_sample.append(rewards_per_func[reward_func_index][sample_index])
                rewards_per_sample.append(rewards_per_func_per_sample)

            reward_list: List[List[float]] = []
            for rewards in rewards_per_sample:
                weighted_reward: List[float] = calculate_weighted_reward(rewards, weights)
                reward_list.append(weighted_reward)

            return reward_list
    
        # except Exception as e:
        #     print(f"Error in reward_weighted_combo:")
        #     print(e)
        #     # return [[0.0] * len(output) for output in outputs]
        #     tokens = 

        # last_token_reward = np.interp(weighted_reward,
        #             (-1, 1),
        #             (-0.1, 0.9))
        # tok_score[-1] = last_token_reward

        # # needed_func_names = count_used_func_names(expected_code)
        # # get keys of count_used_func_names(expected_code) with values > 0
        # needed_func_names = [k for k, v in count_used_func_names(expected_code).items() if v > 0]
        # dense_rewards = reward_substring_matches(response, encodings, needed_func_names, 0.1)

        # # add dense_rewards to tok_score
        # for i in range(len(toks)):
        #     tok_score[i] += dense_rewards[i]

    return reward_weighted_combo
