# inference_gpt2.py

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Step 1: Load the Fine-tuned GPT-2 Model and Tokenizer
model_name_or_path = "gpt2/finetuned_gpt2_epoch3"  # Replace with your model's path if different
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

# Ensure pad_token is defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set model to evaluation mode
model.eval()

# Step 2: Load the Dataset
input_file = "data/public.jsonl"  # Update the path if necessary
output_file = "predictions.jsonl"  # The file to save the results

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

    # Create the input prompt as per training
    # For example: "<BOS><maintext>\nTL;DR:"
    prompt_text = f"{tokenizer.bos_token}{maintext}\nTL;DR:"

    # Tokenize the prompt
    inputs = tokenizer.encode(
        prompt_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )

    # Move inputs to the same device as the model
    inputs = inputs.to(device)

    # Generate the title
    outputs = model.generate(
        inputs,
        max_length=inputs.shape[1] + 64,  # Allow up to 64 tokens for the generated title
        num_beams=5,  # Adjust for desired quality vs. speed
        early_stopping=True,
        no_repeat_ngram_size=2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the generated title by removing the prompt from the output
    # Find the position of "TL;DR:"
    tl_dr_index = generated_text.find("TL;DR:")
    if tl_dr_index != -1:
        title = generated_text[tl_dr_index + len("TL;DR:"):].strip()
    else:
        # If "TL;DR:" is not found, attempt to extract the title
        title = generated_text[len(maintext):].strip()

    # Prepare the result
    result = {
        "title": title,
        "id": id_,
    }
    results.append(result)

# Step 4: Save the Results
with open(output_file, 'w', encoding='utf-8') as f:
    for result in results:
        json_line = json.dumps(result, ensure_ascii=False)
        f.write(json_line + '\n')

print(f"Predictions saved to {output_file}")