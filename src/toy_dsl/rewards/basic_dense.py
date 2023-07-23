from typing import List

from lang import Interpreter

interpreter = Interpreter()

EOS_TOKEN = "<|endoftext|>"

def reward_basic_dense(samples: List[str], prompts: List[str], outputs: List[str], tokenizer, **kwargs) -> List[List[float]]:
    reward_list: List[List[float]] = []
    # for sample in samples:
    for sample, prompt, generated in zip(samples, prompts, outputs):
        code = sample.split("Function:")[1].strip()

        if code.endswith(EOS_TOKEN):
            code = code[:-len(EOS_TOKEN)]

        output = eval(sample.split("Output:")[1].strip().split("Function:")[0].strip())
        interpreted_output = interpreter(code)

        final_reward = 0.0
        if interpreted_output == "ERROR":
            # If the code is unparsable, we give it a negative reward.
            # reward_list.append(-1)
            final_reward = -1
        else:
            # if the code is parseable
            if output == interpreted_output:
                # if the output is correct, we give it a positive reward.
                # reward_list.append(1)
                final_reward = 1
            else:
                # if the output is incorrect, we give it a negative reward.
                # reward_list.append(-0.5)
                final_reward = -0.5

        # rewards_per_token: List[float] = [0.0] * len(tokenizer.tokenize(code, add_special_tokens=False))
        num_tokens = len(tokenizer.tokenize(generated, add_special_tokens=False))
        rewards_per_token: List[float] = [0.0] * num_tokens
        rewards_per_token[-1] = final_reward
        reward_list.append(rewards_per_token)

    return reward_list
