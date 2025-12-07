import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import re # <-- New import for cleaning


# TEXT CLEANING FUNCTION (MUST MATCH data.py)

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # optional for Nepali Romanized
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load saved model and tokenizer
MODEL_DIR = "saved_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Mapping from label IDs to names (change if your training labels differ)
id2label = {0: "Non-Hate Speech", 1: "Hate Speech"}

def classify_text(raw_text):
    # Apply the exact same cleaning as during training
    text = clean_text(raw_text) # <-- Apply the cleaning function!

    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        label_idx = int(probs.argmax())
        confidence = float(probs[label_idx])
        label = id2label.get(label_idx, str(label_idx))

    return {"label": label, "confidence": confidence, "probs": probs.tolist()}

# Interactive testing
if __name__ == "__main__":
    print("=== Hate Speech Detection Test ===")
    while True:
        text = input("\nEnter text (or 'exit' to quit): ")
        if text.lower() == "exit":
            break
        result = classify_text(text)
        print(f"Prediction: {result['label']} | Confidence: {result['confidence']:.4f}")