from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import time


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

def load_model(precision):
    if precision == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype = torch.float16,
            device_map = "cpu",
            low_cpu_mem_usage = False,
        )
    elif precision == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config = BitsAndBytesConfig(load_in_8bit = True),
            device_map = "cpu",
        )
    elif precision == "int4":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_compute_dtype = torch.float16,
            ),
            device_map = "cpu",
        )
    else:
        raise ValueError(f"Unknown Precision: {precision}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return model, tokenizer

def runner(model, tokenizer, prompt):    
    message = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(message, add_generation_prompt = True, return_tensors = "pt").to(model.device)

    start_time = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens = 200)
    end_time = time.time()

    elapsed_time = end_time - start_time

    input_length = inputs["input_ids"].shape[1]
    decoded_text = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens = True)

    return decoded_text, elapsed_time

model, tokenizer = load_model("fp16")
response, elapsed = runner(model, tokenizer, "What is the capital of France?")
print(f"Response: {response}")
print(f"Elapsed: {elapsed:.2f} seconds")
