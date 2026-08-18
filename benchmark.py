import json
from loader import load_model, measure_memory_usage, runner
import gc

model, tokenizer = None, None
def run_benchmark(precisions, questions):
    results = []
    model, tokenizer = None, None

    for precision in precisions:
        if model is not None:
            del model, tokenizer
            gc.collect()
        model, tokenizer, memory_used = measure_memory_usage(precision)
        print(f"Loaded {precision}, memory: {memory_used / (1024**3):.2f} GB")

        for question in questions:
            response, elapsed_time, num_tokens, tokens_per_second = runner(model, tokenizer, question["question"])
            print(f"  Q{question['id']}: {elapsed_time:.2f}s")
            record = {
                "precision": precision,
                "question_id": question["id"],
                "question": question["question"],
                "type": question["type"],
                "expected_answer": question["expected_answer"],
                "response": response,
                "elapsed_time": elapsed_time,
                "num_tokens": num_tokens,
                "tokens_per_second": tokens_per_second,
                "memory_used": memory_used
            }
            results.append(record)

    return results


if __name__ == "__main__":
    with open("eval/questions.json", "r") as f:
        questions = json.load(f)

    precisions = ["fp16", "int8", "int4"]

    results = run_benchmark(precisions, questions)

    with open("results/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)