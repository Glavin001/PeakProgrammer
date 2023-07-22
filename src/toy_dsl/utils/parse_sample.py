EOS_TOKEN = "<|endoftext|>"

def parse_sample(sample: str):
    code = sample.split("Function:")[1].strip()

    if code.endswith(EOS_TOKEN):
        code = code[:-len(EOS_TOKEN)]

    output = eval(sample.split("Output:")[1].strip().split("Function:")[0].strip())
    return {
        "code": code,
        "output": output,
    }
