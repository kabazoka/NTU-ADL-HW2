# Import necessary libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, TrainerCallback
from datasets import load_dataset
import numpy as np
import nltk
from tw_rouge import get_rouge
import pandas as pd
import matplotlib.pyplot as plt

# Download NLTK data files (if not already done)
nltk.download('punkt')

# Define the MetricsLoggerCallback
class MetricsLoggerCallback(TrainerCallback):
    def __init__(self):
        self.metrics = {
            'step': [],
            'epoch': [],
            'train_loss': [],
            'eval_loss': [],
            'rouge_1_f': [],
            'rouge_2_f': [],
            'rouge_l_f': [],
            'avg_rouge_f': [],
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if 'loss' in logs:
            self.metrics['step'].append(state.global_step)
            self.metrics['epoch'].append(state.epoch)
            self.metrics['train_loss'].append(logs['loss'])
            # Initialize eval_loss with None
            self.metrics['eval_loss'].append(None)
            # Initialize Rouge scores with None
            self.metrics['rouge_1_f'].append(None)
            self.metrics['rouge_2_f'].append(None)
            self.metrics['rouge_l_f'].append(None)
            self.metrics['avg_rouge_f'].append(None)
        if 'eval_loss' in logs:
            self.metrics['step'].append(state.global_step)
            self.metrics['epoch'].append(state.epoch)
            self.metrics['train_loss'].append(None)
            self.metrics['eval_loss'].append(logs['eval_loss'])
            # Extract Rouge scores
            rouge_1_f = logs.get('eval_rouge-1_f', None)
            rouge_2_f = logs.get('eval_rouge-2_f', None)
            rouge_l_f = logs.get('eval_rouge-l_f', None)
            # Compute average Rouge F-score
            avg_rouge_f = None
            if rouge_1_f is not None and rouge_2_f is not None and rouge_l_f is not None:
                avg_rouge_f = (rouge_1_f + rouge_2_f + rouge_l_f) / 3.0
            self.metrics['rouge_1_f'].append(rouge_1_f)
            self.metrics['rouge_2_f'].append(rouge_2_f)
            self.metrics['rouge_l_f'].append(rouge_l_f)
            self.metrics['avg_rouge_f'].append(avg_rouge_f)

# Step 2: Load the Pre-trained Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

# Step 3: Load the Dataset
data_files = {"train": "data/train.jsonl"}
raw_datasets = load_dataset('json', data_files=data_files)

# Split the train dataset into training and validation sets
split_datasets = raw_datasets["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)

# Step 4: Preprocess the Data
def preprocess_function(examples):
    inputs = examples["maintext"]
    targets = examples["title"]
    model_inputs = tokenizer(
        inputs,
        max_length=256,  # Truncate inputs to 256 tokens
        truncation=True,
    )

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

# Step 5: Set Up Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=1000,
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Adjust based on your GPU memory
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,  # Accumulate gradients over 8 steps
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,  # You can increase this
    predict_with_generate=True,
    fp16=False,  # Keep as False due to known issues with T5 models
    optim="adafactor",  # Use Adafactor optimizer
)

# Step 6: Define Evaluation Metrics using tw_rouge
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode the predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute Rouge scores using tw_rouge
    result = get_rouge(decoded_preds, decoded_labels, avg=True, ignore_empty=True)

    # Flatten the nested dictionary
    flattened_result = {}
    for rouge_type, scores in result.items():
        for metric, value in scores.items():
            if metric == 'f':
                key = f"{rouge_type}_{metric}"
                flattened_result[f"eval_{key}"] = value * 100  # Keep as float, no rounding
    return flattened_result

# Step 7: Initialize the Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,  # Ensures dynamic padding
)

# Initialize the metrics logger
metrics_logger = MetricsLoggerCallback()

# Step 8: Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],  # Use the validation split
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[metrics_logger],  # Include the callback here
)

# Step 9: Start Training
trainer.train()

# Step 10: Save the Model
trainer.save_model("finetuned_mt5_small")

# Step 11: Access Collected Metrics
metrics_df = pd.DataFrame(metrics_logger.metrics)

# Save metrics to a CSV file (optional)
metrics_df.to_csv('plot/training_metrics.csv', index=False)

# Plot training and validation loss
train_loss = metrics_df.dropna(subset=['train_loss'])
eval_loss = metrics_df.dropna(subset=['eval_loss'])

plt.figure(figsize=(10, 6))
plt.plot(train_loss['step'], train_loss['train_loss'], label='Training Loss')
plt.plot(eval_loss['step'], eval_loss['eval_loss'], label='Validation Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Time')
plt.legend()
plt.grid()

# Save the plot instead of showing it
plt.savefig('plot/training_validation_loss.png')
plt.close()

# Plot average Rouge F-score
eval_metrics = metrics_df.dropna(subset=['avg_rouge_f'])

plt.figure(figsize=(10, 6))
plt.plot(eval_metrics['step'], eval_metrics['avg_rouge_f'], label='Average Rouge F-score')
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
plt.plot(eval_metrics['step'], eval_metrics['rouge_1_f'], label='Rouge-1 F-score')
plt.plot(eval_metrics['step'], eval_metrics['rouge_2_f'], label='Rouge-2 F-score')
plt.plot(eval_metrics['step'], eval_metrics['rouge_l_f'], label='Rouge-L F-score')
plt.xlabel('Step')
plt.ylabel('Rouge F-score (%)')
plt.title('Rouge F-scores over Time')
plt.legend()
plt.grid()

# Save the plot instead of showing it
plt.savefig('plot/rouge_fscores.png')
plt.close()