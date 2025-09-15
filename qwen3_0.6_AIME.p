# ===============================
# Install dependencies (only first time)
# ===============================
# !pip install vllm==0.10.2 pandas

import pandas as pd
import re
from vllm import LLM, SamplingParams

# ===============================
# Load your dataset
# CSV must have: problem, answer
# ===============================
csv_path = "./aime_dataset.csv"  # update path if needed
df = pd.read_csv(csv_path)


print(f"Loaded {len(df)} problems from {csv_path}")

# ===============================
# Load the Qwen model locally
# ===============================
model_path = "./Qwen3-8B"
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=4096  # adjust based on your VRAM
)

sampling_params = SamplingParams(
    temperature=0.0,    # deterministic output
    max_tokens=1024
)

# ===============================
# Evaluation loop
# ===============================
correct = 0
total = 0

for idx, row in df.iterrows():
    problem, expected = row["problem"], str(row["answer"]).strip()

    prompt = f"""Solve the following math problem step by step and give the final numeric answer only.

Problem: {problem}

Answer:"""

    outputs = llm.generate([prompt], sampling_params)
    pred = outputs[0].outputs[0].text.strip()

    # Extract only last integer/number in prediction
    numbers = re.findall(r"-?\d+", pred)
    pred_number = numbers[-1] if numbers else pred

    is_correct = (pred_number == expected)
    correct += int(is_correct)
    total += 1

    print(f"\nProblem {idx+1}: {problem}")
    print(f"Expected: {expected}, Got: {pred_number}, Correct: {is_correct}")

# ===============================
# Final accuracy
# ===============================
accuracy = correct / total if total > 0 else 0
print(f"\nFinal Accuracy: {accuracy:.2%} ({correct}/{total})")
