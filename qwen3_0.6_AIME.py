# ===============================
# Install dependencies (only first time)
# ===============================
# !pip install vllm==0.10.2 pandas tqdm

import re
import pandas as pd
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

# ===============================
# Load your dataset
# ===============================
csv_path = "./aime_dataset.csv"  # update path if needed
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} problems from {csv_path}")

# ===============================
# Load the Qwen model locally
# ===============================
model_path = "./Qwen3-8B"

sampling_params = SamplingParams(
    temperature=0.0,   # deterministic output
    max_tokens=512
)

# ===============================
# Helper: smarter integer extractor
# ===============================
def extract_final_int(text: str):
    """
    Extract the integer that appears right after the marker 'Final Answer:'.
    """
    if not text:
        return None

    # Look for 'Final Answer:' marker
    match = re.search(r"Final Answer:\s*(-?\d+)", text)
    if match:
        return int(match.group(1))

    # Fallback: last integer in text
    numbers = re.findall(r"-?\d+", text)
    if numbers:
        return int(numbers[-1])

    return None

    if not text:
        return None

    # Look for explicit answer markers first
    match = re.search(
        r"(?:Answer|Final Answer|Therefore[, ]*the answer is)[:\s]*(-?\d+)",
        text,
        re.I
    )
    if match:
        return int(match.group(1))

    # Fallback: take last integer in text
    numbers = re.findall(r"-?\d+", text)
    if numbers:
        return int(numbers[-1])

    return None

# ===============================
# Evaluation loop
# ===============================
results = []
correct = 0
total = 0

# Create model (no context manager)
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=10000
)

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
    problem = str(row["problem"]).strip()
    expected = int(row["answer"])  # guaranteed integer

    prompt = f"""Solve the following math problem **step by step**. 
At the end, write the **final numeric answer only** on a new line after the exact phrase:

Final Answer:

Problem: {problem}

Final Answer:"""


    # run model
    outputs = llm.generate([prompt], sampling_params)
    first_output = next(iter(outputs))
    pred_text = first_output.outputs[0].text.strip()

    # extract integer
    pred_int = extract_final_int(pred_text)

    is_correct = (pred_int == expected)
    correct += int(is_correct)
    total += 1

    # Save result
    results.append({
        "problem": problem,
        "expected": expected,
        "model_output": pred_text,
        "predicted": pred_int,
        "correct": is_correct
    })

# ===============================
# Save incorrect cases
# ===============================
errors = [r for r in results if not r["correct"]]
if errors:
    pd.DataFrame(errors).to_csv("errors.csv", index=False)
    print(f"\nSaved {len(errors)} incorrect cases to errors.csv")

# ===============================
# Final accuracy
# ===============================
accuracy = correct / total if total > 0 else 0
print(f"\nFinal Accuracy: {accuracy:.2%} ({correct}/{total})")

