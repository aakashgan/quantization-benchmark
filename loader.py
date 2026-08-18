from mlx_lm import load, generate, convert
import mlx.core as mx
import time


def load_model(precision):
    paths = {
        "fp16": "./llama-fp16",
        "int8": "./llama-int8",
        "int4": "./llama-int4",
    }
    if precision not in paths:
        raise ValueError(f"Unknown precision: {precision}")

    model, tokenizer = load(paths[precision])
    return model, tokenizer

def runner(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    start_time = time.time()
    response = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=200, verbose=False)
    end_time = time.time()
    elapsed_time = end_time - start_time

    num_tokens = len(tokenizer.encode(response))
    tokens_per_second = num_tokens / elapsed_time

    return response, elapsed_time, num_tokens, tokens_per_second

def measure_memory_usage(precision):
    mx.reset_peak_memory()
    
    model, tokenizer = load_model(precision)
    
    mx.eval(model.parameters())
    
    memory_used = mx.get_active_memory()
    
    return model, tokenizer, memory_used

