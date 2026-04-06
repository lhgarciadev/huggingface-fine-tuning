# Lab 4: Training with Trainer API

In this lab, you will learn how to train transformer models using the Hugging Face Trainer API. You'll build a text classification model, run inference, and compare different pre-trained models.

## Learning Objectives

By the end of this lab, you will be able to:

- Configure and instantiate the Trainer API
- Train models on custom datasets
- Perform inference with trained models
- Compare different model architectures
- Extract predictions and confidence scores

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install transformers datasets pandas torch numpy scikit-learn
```

## Key Concepts

- **Trainer API**: High-level interface for training transformer models
- **TrainingArguments**: Configuration for training hyperparameters
- **AutoModelForSequenceClassification**: Pre-trained model for classification tasks
- **AutoTokenizer**: Tokenizer matched to a specific model
- **Inference**: Using a trained model to make predictions

## Lab Exercises

### Exercise 1: Basic Training Setup

Navigate to the [examples/training](../examples/training/) directory.

1. Study [classifier.py](../examples/training/classifier.py):

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd

# 1. Load your data
df = pd.read_csv('status.csv')
dataset = Dataset.from_pandas(df)

# 2. Split into train/test
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
print(f"Train: {len(split_dataset['train'])} examples")
print(f"Test: {len(split_dataset['test'])} examples")

# 3. Load pre-trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2  # Binary classification
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

2. Run the training:

```bash
cd examples/training
python classifier.py
```

### Exercise 2: Tokenization for Training

Understand how to prepare data for the Trainer:

```python
def tokenize_and_label(examples):
    # Tokenize the text
    tokenized = tokenizer(
        examples['status'],
        truncation=True,
        padding=True
    )
    # Add labels for classification
    tokenized['labels'] = examples['Blankets_Creek']
    return tokenized

# Apply to entire dataset
tokenized_datasets = split_dataset.map(tokenize_and_label, batched=True)
```

### Exercise 3: Training Configuration

Learn the key TrainingArguments parameters:

```python
training_args = TrainingArguments(
    output_dir="./trail_classifier",    # Where to save the model
    num_train_epochs=3,                  # Number of training passes
    per_device_train_batch_size=8,       # Batch size for training
    per_device_eval_batch_size=8,        # Batch size for evaluation
    logging_steps=10,                    # Log every N steps
    eval_strategy="epoch",               # Evaluate after each epoch
)
```

### Exercise 4: Creating and Running the Trainer

```python
# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Train the model
print("Starting training...")
trainer.train()
print("Training complete!")

# Save the model and tokenizer
trainer.save_model("./trail_classifier")
tokenizer.save_pretrained("./trail_classifier")
```

### Exercise 5: Running Inference

1. Study [inference.py](../examples/inferencing/inference.py):

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the trained model
model = AutoModelForSequenceClassification.from_pretrained("./trail_classifier")
tokenizer = AutoTokenizer.from_pretrained("./trail_classifier")

# Setup device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def predict_trail_status(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = outputs.logits.argmax().item()
    confidence = torch.softmax(outputs.logits, dim=1)[0][prediction].item()
    return prediction, confidence

# Test predictions
test_cases = [
    "trails are open today",
    "park closed due to storm",
    "blankets creek is open",
]

for text in test_cases:
    prediction, confidence = predict_trail_status(text)
    status = "OPEN" if prediction == 1 else "CLOSED"
    print(f"'{text}' -> {status} ({confidence:.1%} confidence)")
```

2. Run inference on your trained model from the repo root:

```bash
uv run python examples/inferencing/inference.py trail_classifier
```

The script resolves local model folders from common example directories, so it works whether you trained from `examples/training/` or from one of the model-specific example folders.

### Exercise 6: Comparing Different Models

Navigate to the [examples/models](../examples/models/) directory.

1. Compare different pre-trained models:

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| bert-base-uncased | 110M params | Fast | Good |
| roberta-large | 355M params | Slower | Better |
| microsoft/deberta-v3-large | 400M params | Slowest | Best |

2. Study the model comparison files:
   - [classifier.py](../examples/models/classifier.py) - BERT base
   - [classifier-large.py](../examples/models/classifier-large.py) - RoBERTa large
   - [classifier-msft.py](../examples/models/classifier-msft.py) - Microsoft DeBERTa

3. Train with a different model:

```python
# Change just the model name
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-large",  # Different model
    num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
```

### Exercise 7: Evaluating Model Performance

Add metrics to track model performance:

```python
import numpy as np
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,  # Add metrics
)
```

## Challenge

1. Train a model on the trail status dataset
2. Run inference on 10 new test cases you create
3. Train a second model using a different pre-trained base (e.g., RoBERTa)
4. Compare the predictions and confidence scores between models
5. Document which model performs better on your test cases

## Summary

In this lab, you learned how to:
- Set up the Trainer API with proper configuration
- Tokenize and prepare data for training
- Train a text classification model
- Save and load trained models
- Run inference and extract predictions with confidence scores
- Compare different pre-trained model architectures

## Next Steps

Continue to [Lab 5: Advanced Training and Callbacks](./lab-5.md) to learn optimization techniques, metrics, and debugging strategies.
