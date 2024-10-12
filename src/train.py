# Import necessary libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
import numpy as np
import nltk
from tw_rouge import get_rouge
import pandas as pd
import matplotlib.pyplot as plt
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Download NLTK data files
nltk.download('punkt')

# Initialize the Accelerator
accelerator = Accelerator(mixed_precision="fp16")  # Set fp16=true

# Step 1: Load the Pre-trained Model and Tokenizer
model_name_or_path = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

# Prepare model with Accelerator
model = accelerator.prepare(model)

# Step 2: Load the Dataset
data_files = {"train": "data/train.jsonl"}
raw_datasets = load_dataset('json', data_files=data_files)

# Split the train dataset into training and validation sets
split_datasets = raw_datasets["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)

# Step 3: Preprocess the Data
def preprocess_function(examples):
    inputs = examples["maintext"]
    targets = examples["title"]
    model_inputs = tokenizer(
        inputs,
        max_length=256,  # Truncate inputs to 256 tokens
        truncation=True,
    )

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=64,  # Truncate targets to 64 tokens
            truncation=True,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)

# Step 4: Prepare DataLoaders
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8  # Adjust based on your GPU memory
)

eval_dataloader = DataLoader(
    tokenized_datasets["test"],
    collate_fn=data_collator,
    batch_size=8  # Adjust based on your GPU memory
)

# Prepare the data loaders using Accelerator
train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

# Step 5: Set Up the Optimizer and Scheduler
from transformers.optimization import Adafactor, AdafactorSchedule

optimizer = Adafactor(
    model.parameters(),
    scale_parameter=True,
    relative_step=True,
    warmup_init=True,
    lr=None,  # None for relative step with warmup
)

# Prepare the optimizer with Accelerator
optimizer = accelerator.prepare(optimizer)

# Learning rate scheduler (optional)
lr_scheduler = AdafactorSchedule(optimizer)

# Step 6: Training Loop
num_train_epochs = 10  # Adjust as needed
gradient_accumulation_steps = 8
logging_steps = 1000

# Calculate total training steps
total_steps = (len(train_dataloader) // gradient_accumulation_steps) * num_train_epochs

progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)

# Initialize metrics logging
training_loss = []
eval_loss_list = []
rouge_scores = {'rouge_1_f': [], 'rouge_2_f': [], 'rouge_l_f': [], 'avg_rouge_f': []}
global_step = 0

for epoch in range(num_train_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)

        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
            optimizer.step()
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
    all_predictions = []
    all_labels = []

    for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch+1}", disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            outputs = model(**batch)
            eval_loss = outputs.loss
            eval_losses.append(accelerator.gather(eval_loss.repeat(batch["input_ids"].size(0))))

            # Generate predictions
            generated_tokens = accelerator.unwrap_model(model).generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=64,
                num_beams=5,
            )

            labels = batch["labels"]

            # Replace -100 in the labels as we can't decode them
            labels = labels.clone()
            labels[labels == -100] = tokenizer.pad_token_id

            # Gather predictions and labels
            generated_tokens, labels = accelerator.gather((generated_tokens, labels))

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_predictions.extend(decoded_preds)
            all_labels.extend(decoded_labels)

    # Compute evaluation metrics
    eval_loss = torch.cat(eval_losses).mean().item()
    eval_loss_list.append((global_step, eval_loss))

    result = get_rouge(all_predictions, all_labels, avg=True, ignore_empty=True)

    # Compute average Rouge F-score
    avg_rouge_f = (
        result['rouge-1']['f'] +
        result['rouge-2']['f'] +
        result['rouge-l']['f']
    ) / 3.0

    rouge_scores['rouge_1_f'].append((global_step, result['rouge-1']['f'] * 100))
    rouge_scores['rouge_2_f'].append((global_step, result['rouge-2']['f'] * 100))
    rouge_scores['rouge_l_f'].append((global_step, result['rouge-l']['f'] * 100))
    rouge_scores['avg_rouge_f'].append((global_step, avg_rouge_f * 100))

    if accelerator.is_main_process:
        print(f"\nEpoch {epoch+1} Evaluation:")
        print(f"Eval Loss: {eval_loss}")
        print(f"Rouge-1 F-score: {result['rouge-1']['f'] * 100:.2f}")
        print(f"Rouge-2 F-score: {result['rouge-2']['f'] * 100:.2f}")
        print(f"Rouge-L F-score: {result['rouge-l']['f'] * 100:.2f}")
        print(f"Average Rouge F-score: {avg_rouge_f * 100:.2f}")

        # Save the model at the end of each epoch
        output_dir = f"mt5/finetuned_mt5_epoch{epoch+1}"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

# Step 7: Metrics Logging and Plotting
# Convert metrics to DataFrames
metrics_df = pd.DataFrame({
    'step': [step for step, _ in training_loss],
    'train_loss': [loss for _, loss in training_loss],
})

eval_metrics_df = pd.DataFrame({
    'step': [step for step, _ in eval_loss_list],
    'eval_loss': [loss for _, loss in eval_loss_list],
    'rouge_1_f': [score for _, score in rouge_scores['rouge_1_f']],
    'rouge_2_f': [score for _, score in rouge_scores['rouge_2_f']],
    'rouge_l_f': [score for _, score in rouge_scores['rouge_l_f']],
    'avg_rouge_f': [score for _, score in rouge_scores['avg_rouge_f']],
})

# Save metrics to CSV
os.makedirs('plot', exist_ok=True)
metrics_df.to_csv('plot/training_metrics.csv', index=False)
eval_metrics_df.to_csv('plot/eval_metrics.csv', index=False)

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
plt.savefig('plot/training_validation_loss.png')
plt.close()

# Plot average Rouge F-score
plt.figure(figsize=(10, 6))
plt.plot(eval_metrics_df['step'], eval_metrics_df['avg_rouge_f'], label='Average Rouge F-score')
plt.xlabel('Step')
plt.ylabel('Average Rouge F-score (%)')
plt.title('Average Rouge F-score over Time')
plt.legend()
plt.grid()

# Save the plot instead of showing it
plt.savefig('plot/average_rouge_fscore.png')
plt.close()

# Plot Rouge F-scores
plt.figure(figsize=(10, 6))
plt.plot(eval_metrics_df['step'], eval_metrics_df['rouge_1_f'], label='Rouge-1 F-score')
plt.plot(eval_metrics_df['step'], eval_metrics_df['rouge_2_f'], label='Rouge-2 F-score')
plt.plot(eval_metrics_df['step'], eval_metrics_df['rouge_l_f'], label='Rouge-L F-score')
plt.xlabel('Step')
plt.ylabel('Rouge F-score (%)')
plt.title('Rouge F-scores over Time')
plt.legend()
plt.grid()

# Save the plot instead of showing it
plt.savefig('plot/rouge_fscores.png')
plt.close()