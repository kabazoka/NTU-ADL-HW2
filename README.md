# Text Summarization with Fine-tuned mT5 Model

This project involves fine-tuning the multilingual T5 (mT5) model for the task of text summarization—specifically, generating titles from main text content. The repository includes scripts for training the model and performing inference on new data.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Performing Inference](#performing-inference)
- [Model Details](#model-details)
- [Data Preprocessing](#data-preprocessing)
- [Hyperparameters](#hyperparameters)
- [Generation Strategies](#generation-strategies)
- [Results and Evaluation](#results-and-evaluation)
- [Learning Curves](#learning-curves)
- [Acknowledgements](#acknowledgements)

---

## Introduction

This project fine-tunes the "google/mt5-small" model—a multilingual variant of the T5 model—for text summarization tasks. The goal is to generate concise and informative titles from longer pieces of text.

---

## Features

- **Fine-tuning Script:** Customizable training script with mixed-precision support using Hugging Face Accelerate.
- **Inference Script:** Easy-to-use script for generating summaries from new data.
- **Generation Strategies:** Supports various text generation strategies, including Beam Search.
- **Metrics Logging:** Tracks training loss and ROUGE scores during training.
- **Visualization:** Generates plots for loss curves and ROUGE scores over time.

---

## Requirements

- **Python 3.6+**
- **PyTorch**
- **Transformers**
- **Datasets**
- **NLTK**
- **tqdm**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Accelerate**
- **tw-rouge**

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/mt5-summarization.git
   cd mt5-summarization
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not provided, install the packages manually:

   ```bash
   pip install torch transformers datasets nltk tqdm numpy pandas matplotlib accelerate tw-rouge
   ```

4. **Download NLTK Data**

   ```python
   import nltk
   nltk.download('punkt')
   ```

---

## Usage

### Training the Model

1. **Prepare Your Dataset**

   - Place your training data in the `data/` directory.
   - The training data should be in JSON Lines (`.jsonl`) format with at least two fields: `"maintext"` and `"title"`.

     Example:

     ```json
     {"maintext": "This is the main text content.", "title": "Generated Title"}
     ```

2. **Run the Training Script**

   ```bash
   python src/train_mt5.py
   ```

   - Ensure that the paths to the training data in `train.py` are correct.
   - Adjust hyperparameters as needed within the script.

### Performing Inference

Use the provided `run.sh` script to perform inference on new data.

```bash
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```

- **`/path/to/input.jsonl`**: Path to the input JSON Lines file containing the data to summarize.
- **`/path/to/output.jsonl`**: Path where the output with generated summaries will be saved.

**Input File Format**

The input JSON Lines file should contain entries with at least `"id"` and `"maintext"` fields.

Example:

```json
{"id": "sample1", "maintext": "This is the main text to be summarized."}
{"id": "sample2", "maintext": "Another piece of text for title generation."}
```

**Output File Format**

The output will be a JSON Lines file with `"id"` and the generated `"title"`.

Example:

```json
{"title": "Generated title for sample1", "id": "sample1"}
{"title": "Generated title for sample2", "id": "sample2"}
```

---

## Model Details

- **Model Architecture**: mT5-small, a multilingual Text-to-Text Transfer Transformer.
- **Encoder-Decoder Structure**:
  - **Encoder**: Processes the input text and creates contextual embeddings.
  - **Decoder**: Generates the output text (title) using the encoder's embeddings and its own outputs.
- **Application in Summarization**:
  - The model learns to map from main text content to concise titles, effectively summarizing the input.

---

## Data Preprocessing

- **Tokenization**:
  - Inputs (`maintext`) and targets (`title`) are tokenized using the model's tokenizer.
- **Truncation and Padding**:
  - **Inputs**: Truncated to a maximum length of **256 tokens**.
  - **Targets**: Truncated to a maximum length of **64 tokens**.
- **Label Preparation**:
  - Tokenized target sequences are assigned to the `"labels"` key for loss computation.
- **Data Cleaning**:
  - Assumes the dataset is clean; no additional preprocessing steps like removing special characters are applied.

---

## Hyperparameters

### Training Hyperparameters

- **Number of Epochs**: `num_train_epochs = 10`
- **Batch Size**: `batch_size = 8` (effective batch size is 64 due to gradient accumulation)
- **Gradient Accumulation Steps**: `gradient_accumulation_steps = 8`
- **Optimizer**: Adafactor with relative step size and warmup
- **Learning Rate**: Controlled by Adafactor (`lr = None`)
- **Logging Steps**: `logging_steps = 1000`

### Generation Parameters

- **Beam Search**:
  - `num_beams = 5`
- **Maximum Generation Length**:
  - `max_length = 128`
- **Early Stopping**:
  - `early_stopping = True`

---

## Generation Strategies

### Beam Search (Final Strategy)

- **Description**:
  - At each decoding step, keeps the top `num_beams` most probable sequences.
- **Parameters**:
  - `num_beams = 5`
- **Advantages**:
  - Provides a balance between exploration of multiple hypotheses and computational efficiency.
  - Consistently produced the best ROUGE scores in experiments.

### Other Strategies (Experimented)

- **Greedy Search**:
  - Selects the most probable token at each step.
- **Top-k Sampling**:
  - Samples from the top `k` tokens at each step.
- **Top-p (Nucleus) Sampling**:
  - Samples from the smallest set of tokens with a cumulative probability exceeding `p`.
- **Temperature Adjustment**:
  - Scales the logits to adjust the randomness in sampling.

**Note**: Beam Search with `num_beams = 5` was chosen as the final strategy based on experimental results.

---

## Results and Evaluation

### Evaluation Metrics

- **ROUGE-1 F-score**
- **ROUGE-2 F-score**
- **ROUGE-L F-score**
- **Average ROUGE F-score**

### Observations

- **Beam Search** with `num_beams = 5` yielded the best performance.
- **Temperature Adjustment** to `temperature = 0.7` was tested but not included in the final strategy.

---

### Available Plots

- **Training and Validation Loss**: Shows the loss over training steps.
  - Saved as `plot/training_validation_loss.png`
- **Average ROUGE F-score**: Visualizes the average ROUGE score over time.
  - Saved as `plot/average_rouge_fscore.png`
- **ROUGE F-scores**: Individual plots for ROUGE-1, ROUGE-2, and ROUGE-L scores.
  - Saved as `plot/rouge_fscores.png`

---

## Acknowledgements

- **Hugging Face Transformers**: For providing the mT5 model and tools for NLP tasks.
- **PyTorch**: The deep learning framework used for model training and inference.
- **NLTK**: Used for natural language processing tasks.
- **tw-rouge**: For calculating ROUGE scores during evaluation.
