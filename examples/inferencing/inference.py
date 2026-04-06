"""Run inference with a fine-tuned trail classifier."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        nargs="?",
        default="trail_classifier",
        help="Path to the saved model directory.",
    )
    return parser.parse_args()


args = parse_args()
model_arg = Path(args.model_name)
if model_arg.is_absolute():
    candidate_paths = [model_arg]
else:
    candidate_paths = [
        Path.cwd() / model_arg,
        SCRIPT_DIR / model_arg,
        SCRIPT_DIR.parent / "training" / model_arg,
        SCRIPT_DIR.parent / "loading" / model_arg,
        SCRIPT_DIR.parent / "models" / model_arg,
        SCRIPT_DIR.parent / "custom" / model_arg,
        SCRIPT_DIR.parent / "callback" / model_arg,
        SCRIPT_DIR.parent / "publishing" / model_arg,
    ]

for candidate in candidate_paths:
    if candidate.exists():
        model_path = candidate
        break
else:
    raise FileNotFoundError(
        "Model directory was not found. Tried: " + ", ".join(str(path) for path in candidate_paths)
    )

# 1. Load the trained model
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device}")


# 3. Prediction function
def predict_trail_status(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():  # Don't compute gradients for inference
        outputs = model(**inputs)
    prediction = outputs.logits.argmax().item()
    confidence = torch.softmax(outputs.logits, dim=1)[0][prediction].item()
    return prediction, confidence


# 4. Test it!
test_cases = [
    "trails are open today",
    "park closed due to storm",
    "blankets creek is open",
    "everything is closed",
    "all trails are open yippie!!",
    "blankets creek is closed rope mill is open",
]

print("\nTrail Status Predictions:")
print("=" * 50)
for text in test_cases:
    prediction, confidence = predict_trail_status(text)
    status = "OPEN ✅" if prediction == 1 else "CLOSED ❌"
    print(f"'{text}'")
    print(f"  → {status} ({confidence:.1%} confidence)")
    print()
