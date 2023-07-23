from typing import List

# from lang import Interpreter
# import Interpreter from lang up one directory
from lang import Interpreter

interpreter = Interpreter()

EOS_TOKEN = "<|endoftext|>"

def reward_basic(samples: List[str], prompts: List[str], outputs: List[str], tokenizer, **kwargs) -> List[float]:
    reward_list: List[float] = []
    for sample in samples:
        code = sample.split("Function:")[1].strip()

        if code.endswith(EOS_TOKEN):
            code = code[:-len(EOS_TOKEN)]

        output = eval(sample.split("Output:")[1].strip().split("Function:")[0].strip())
        interpreted_output = interpreter(code)
        if interpreted_output == "ERROR":
            # If the code is unparsable, we give it a negative reward.
            reward_list.append(-1)
        else:
            # if the code is parseable
            if output == interpreted_output:
                # if the output is correct, we give it a positive reward.
                reward_list.append(1)
            else:
                # if the output is incorrect, we give it a negative reward.
                reward_list.append(-0.5)

    return reward_list
