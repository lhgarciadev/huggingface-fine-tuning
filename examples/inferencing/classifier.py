from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent / "loading" / "status.csv"
MODEL_DIR = SCRIPT_DIR / "trail_classifier"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# 1. Load your CSV
df = pd.read_csv(DATA_PATH)

# 2. Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# 3. Split into train/test
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
print(f"Train: {len(split_dataset['train'])} examples")
print(f"Test: {len(split_dataset['test'])} examples")

# 4. Load model for classification (2 labels: open/closed)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,  # Binary classification
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# 5. Tokenize function
# def tokenize_function(examples):
#    return tokenizer(examples['status'], truncation=True, padding=True)
def tokenize_and_label(examples):
    # Tokenize
    tokenized = tokenizer(examples["status"], truncation=True, padding=True)
    # Add labels (convert to list if single value)
    tokenized["labels"] = examples["Blankets_Creek"]
    return tokenized


tokenized_datasets = split_dataset.map(tokenize_and_label, batched=True)

# 6. Training arguments - FIXED: use eval_strategy not evaluation_strategy
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=10,
    eval_strategy="epoch",  # FIXED THIS LINE
)

# 7. Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# 8. Train!
print("Starting training...")
trainer.train()
print("\nTraining complete!")

# 9 Save everything
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)


## 9. Test prediction
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
# sample_text = "trails are open today"
# inputs = tokenizer(sample_text, return_tensors="pt").to(device)
# outputs = model(**inputs)
# prediction = outputs.logits.argmax().item()
#
# print(f"\nPrediction for: '{sample_text}'")
# print(f"Blankets Creek: {'OPEN ✅' if prediction == 1 else 'CLOSED ❌'}")
