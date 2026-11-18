"""Generate, tokenize, and greedily evaluate decimal addition examples."""
import random

import torch

from . import looped


def _operand(rng, digits):
    return rng.randint(0 if digits == 1 else 10 ** (digits - 1), 10 ** digits - 1)


def sample_operands(rng, max_digits):
    if type(max_digits) is not int or max_digits < 1:
        raise ValueError("max_digits must be a positive integer")
    return tuple(_operand(rng, rng.randint(1, max_digits)) for _ in range(2))


def sample_operands_strict3(rng):
    """D4: both operands have exactly three digits."""
    return _operand(rng, 3), _operand(rng, 3)


def format_example(a, b):
    return f"{a}+{b}={a+b}"


def build_test_set(seed=1234, per_length=500):
    rng = random.Random(seed)
    result = []
    for digits in range(1, 7):
        for _ in range(per_length):
            a, b = _operand(rng, digits), _operand(rng, digits)
            result.append(dict(a=a, b=b, answer=a+b))
    return result


def encode_example(tok, a, b):
    # Tokenize each side separately: BPE numbers can contain several tokens.
    # The prompt ends at '=', and EOS is one explicit supervised token.
    if tok.eos_token_id is None:
        raise ValueError("addition requires an EOS token")
    prompt = tok.encode(f"{a}+{b}=", add_special_tokens=False)
    answer = tok.encode(str(a+b), add_special_tokens=False) + [tok.eos_token_id]
    return prompt + answer, [-100] * len(prompt) + answer


def greedy_answer(model, tok, a, b, max_new_tokens=8):
    device = next(model.parameters()).device
    x = torch.tensor([tok.encode(f"{a}+{b}=", add_special_tokens=False)],
                     dtype=torch.long, device=device)
    generated = []
    model.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        for _ in range(max_new_tokens):
            token = looped.forward(model, x, last_only=True).logits[0, -1].argmax().item()
            # Do not decode EOS as part of the numeric answer.
            if token == tok.eos_token_id:
                break
            generated.append(token)
            x = torch.cat((x, x.new_tensor([[token]])), dim=1)
    return tok.decode(generated, skip_special_tokens=True).strip()
