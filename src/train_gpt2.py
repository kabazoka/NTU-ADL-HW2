# Import necessary libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Initialize the Accelerator
accelerator = Accelerator(mixed_precision="fp16")  # Set fp16=true if your hardware supports it

# Step 1: Load the Pre-trained GPT-2 Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained('ckiplab/gpt2-base-chinese')
model = AutoModelForCausalLM.from_pretrained('ckiplab/gpt2-base-chinese')

# Prepare model with Accelerator
model, optimizer = accelerator.prepare(model, None)  # Optimizer will be defined later

# Step 2: Load the Dataset
data_files = {"train": "data/train.jsonl"}
raw_datasets = load_dataset('json', data_files=data_files)

# Split the train dataset into training and validation sets
split_datasets = raw_datasets["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)

# Step 3: Adjust Tokenizer Special Tokens (if necessary)
# GPT-2 tokenizer may not have a pad_token defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Ensure bos_token and eos_token are defined
if tokenizer.bos_token is None:
    tokenizer.bos_token = tokenizer.cls_token if tokenizer.cls_token else tokenizer.eos_token
if tokenizer.eos_token is None:
    tokenizer.eos_token = tokenizer.sep_token if tokenizer.sep_token else tokenizer.pad_token

# Step 4: Preprocess the Data
def preprocess_function(examples):
    inputs = examples["maintext"]
    targets = examples["title"]
    
    # Combine the inputs and targets into a single string with a separator
    # For example: "<BOS><maintext>\nTL;DR:<title><EOS>"
    # Adjust the prompt format as needed
    model_inputs = []
    for inp, tgt in zip(inputs, targets):
        text = f"{tokenizer.bos_token}{inp}\nTL;DR:{tgt}{tokenizer.eos_token}"
        model_inputs.append(text)
    
    # Tokenize the combined text
    tokenized_inputs = tokenizer(
        model_inputs,
        max_length=512,
        truncation=True,
    )
    
    return tokenized_inputs

tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)

# Step 5: Prepare DataLoaders
def data_collator(features):
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
    
    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    # Create labels (same as input_ids)
    labels = input_ids.clone()
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=2  # Adjust based on your GPU memory
)

eval_dataloader = DataLoader(
    tokenized_datasets["test"],
    collate_fn=data_collator,
    batch_size=2  # Adjust based on your GPU memory
)

# Prepare the data loaders using Accelerator
train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

# Step 6: Set Up the Optimizer and Scheduler
from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(), lr=5e-5)

# Prepare the optimizer with Accelerator
model, optimizer = accelerator.prepare(model, optimizer)

# Calculate total training steps
num_train_epochs = 3  # Adjust as needed
gradient_accumulation_steps = 8
logging_steps = 100

total_steps = (len(train_dataloader) // gradient_accumulation_steps) * num_train_epochs

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

# Step 7: Training Loop
progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)

# Initialize metrics logging
training_loss = []
eval_loss_list = []
global_step = 0

for epoch in range(num_train_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
    
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            global_step += 1
    
            # Logging training loss
            if global_step % logging_steps == 0:
                training_loss.append((global_step, loss.item()))
                if accelerator.is_main_process:
                    print(f"Step {global_step}: loss = {loss.item()}")
    
    # Evaluation at the end of each epoch
    model.eval()
    eval_losses = []

    for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch+1}", disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            eval_loss = outputs.loss
            eval_losses.append(accelerator.gather(eval_loss.repeat(batch["input_ids"].size(0))))
    
    # Compute evaluation metrics
    eval_loss = torch.cat(eval_losses).mean().item()
    eval_loss_list.append((global_step, eval_loss))
    
    if accelerator.is_main_process:
        print(f"\nEpoch {epoch+1} Evaluation:")
        print(f"Eval Loss: {eval_loss}")
    
        # Save the model at the end of each epoch
        output_dir = f"gpt2/finetuned_gpt2_epoch{epoch+1}"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

# Step 8: Metrics Logging and Plotting
# Convert metrics to DataFrames
metrics_df = pd.DataFrame({
    'step': [step for step, _ in training_loss],
    'train_loss': [loss for _, loss in training_loss],
})

eval_metrics_df = pd.DataFrame({
    'step': [step for step, _ in eval_loss_list],
    'eval_loss': [loss for _, loss in eval_loss_list],
})

# Save metrics to CSV
os.makedirs('plot', exist_ok=True)
metrics_df.to_csv('plot/gpt_training_metrics.csv', index=False)
eval_metrics_df.to_csv('plot/gpt_eval_metrics.csv', index=False)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(metrics_df['step'], metrics_df['train_loss'], label='Training Loss')
plt.plot(eval_metrics_df['step'], eval_metrics_df['eval_loss'], label='Validation Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Time')
plt.legend()
plt.grid()

# Save the plot instead of showing it
plt.savefig('plot/gpt_training_validation_loss.png')
plt.close()