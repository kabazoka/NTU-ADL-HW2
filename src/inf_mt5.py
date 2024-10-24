# inference.py

import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Check command-line arguments
if len(sys.argv) != 3:
    print("Usage: python inference.py /path/to/input.jsonl /path/to/output.jsonl")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# Step 1: Load the Fine-tuned Model and Tokenizer
model_name_or_path = "mt5/finetuned_mt5"  # Update this path if your model is saved elsewhere
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 2: Load the Dataset
data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line.strip())
        data.append(item)

# Step 3: Perform Inference
results = []
for item in tqdm(data, desc="Generating Titles"):
    maintext = item["maintext"]
    id_ = item["id"]

    # Tokenize the input
    inputs = tokenizer(
        maintext,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )

    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate the title
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,  # Adjust for desired quality vs. speed
        early_stopping=True,
    )

    # Decode the generated title
    title = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Prepare the result
    result = {
        "title": title,
        "id": id_,
    }
    results.append(result)

# Step 4: Save the Results
with open(output_file, 'w', encoding='utf-8') as f:
    for result in results:
        json_line = json.dumps(result, ensure_ascii=True)
        f.write(json_line + '\n')

print(f"Predictions saved to {output_file}")
