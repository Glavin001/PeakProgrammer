from typing import List

# from lang import Interpreter
# import Interpreter from lang up one directory
from lang import Interpreter

interpreter = Interpreter()

EOS_TOKEN = "<|endoftext|>"

MAX_LENGTH = 20 #1024
MAX_TOKEN_COST = 0.2
MIN_TOKEN_COST = 0.01

def make_reward_token_cost(max_length: int = MAX_LENGTH, max_token_cost: float = MAX_TOKEN_COST, min_token_cost: float = MIN_TOKEN_COST):
    def reward_token_cost(samples: List[str], prompts: List[str], outputs: List[str], tokenizer, **kwargs) -> List[List[float]]:
        reward_list = []
        for sample, prompt, output in zip(samples, prompts, outputs):

            # tokens = tokenizer.encode(prompt, output, add_special_tokens=False)
            tokens = tokenizer.tokenize(output, add_special_tokens=False)
            rewards = [0.0] * len(tokens)

            # penalize each subsequent token by its cost
            # cost is exponential in the number of tokens
            # max cost is reached at MAX_LENGTH tokens
            for i in range(len(tokens)):
                # rewards[i] = -MAX_TOKEN_COST * (i/MAX_LENGTH)**2
                # rewards[i] = -0.01
                # rewards[i] = -MAX_TOKEN_COST * (1 - (i/MAX_LENGTH))**2
                # rewards[i] = -MAX_TOKEN_COST * (1 - 2**(i/MAX_LENGTH))**2
                # exponential starting at 0 then peaking highest cost at/after MAX_LENGTH
                rewards[i] = -max_token_cost * (2**(i/max_length) - 1)**2 - min_token_cost

            reward_list.append(rewards)

        return reward_list
    return reward_token_cost