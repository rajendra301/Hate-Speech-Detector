# fixed_train.py
import os
import re
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)

# reproducibility
set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#  Load dataset 
df = pd.read_csv("dataset.csv")
print("Columns in dataset:", df.columns.tolist())

# AUTO-DETECT TEXT , LABEL COLUMNS
TEXT_COLUMN = None
LABEL_COLUMN = None
possible_text_cols = ["text", "tweet", "comment", "sentence", "message"]
possible_label_cols = ["class", "label", "target", "category"]

for col in df.columns:
    if col.lower() in possible_text_cols:
        TEXT_COLUMN = col
    if col.lower() in possible_label_cols:
        LABEL_COLUMN = col

if TEXT_COLUMN is None or LABEL_COLUMN is None:
    raise ValueError(
        f" ERROR: Could not detect text/label columns.\nDetected columns: {df.columns.tolist()}"
    )

print(f"Detected TEXT column  → {TEXT_COLUMN}")
print(f"Detected LABEL column → {LABEL_COLUMN}")

# NORMALIZE LABELS, binary classification (0=non-hate, 1=hate)
def normalize_label(x):
    x = str(x).strip().lower()
    if x in {"1", "hate", "offensive", "true", "yes"}:
        return 1
    return 0

df["label"] = df[LABEL_COLUMN].apply(normalize_label).astype(int)
print("Unique labels after normalization:", df["label"].unique())
if len(df["label"].unique()) < 2:
    raise ValueError("STOP! The dataset only has 1 class after processing. Check your CSV labels.")

# CLEAN TEXT 
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df[TEXT_COLUMN].apply(clean_text)

# TRAIN / VAL / TEST SPLIT (stratified)
train_df, temp_df = train_test_split(
    df[["text", "label"]],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
print("Train label distribution:\n", train_df["label"].value_counts())

# If training set is heavily imbalanced, upsample minority class
min_count = train_df["label"].value_counts().min()
max_count = train_df["label"].value_counts().max()
imbalance_ratio = max_count / min_count if min_count > 0 else None
print(f"Imbalance ratio (major/minor): {imbalance_ratio:.2f}" if imbalance_ratio else "No minority class")

if imbalance_ratio and imbalance_ratio > 1.5:
    # upsample minority class to match majority
    majority_label = train_df["label"].value_counts().idxmax()
    minority_label = train_df["label"].value_counts().idxmin()
    df_major = train_df[train_df["label"] == majority_label]
    df_minor = train_df[train_df["label"] == minority_label]
    df_minor_upsampled = resample(df_minor, replace=True, n_samples=len(df_major), random_state=42)
    train_df = pd.concat([df_major, df_minor_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    print("Performed upsampling on minority class. New distribution:\n", train_df["label"].value_counts())

# MODEL & TOKENIZER
MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Dataset class
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

train_dataset = HateSpeechDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
val_dataset   = HateSpeechDataset(val_df["text"].tolist(),   val_df["label"].tolist(),   tokenizer)
test_dataset  = HateSpeechDataset(test_df["text"].tolist(),  test_df["label"].tolist(),  tokenizer)

# CLASS WEIGHTS 
class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(train_df["label"]),
    y=train_df["label"]
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class weights (0,1):", class_weights.numpy())

# Custom Trainer, we override compute_loss to use weighted CE and override train dataloader to use sampler if desired
class WeightedTrainer(Trainer):
    def __init__(self, *args, use_sampler=False, **kwargs):
        """
        Custom Trainer with optional weighted sampling.
        """
        super().__init__(*args, **kwargs)
        self.use_sampler = use_sampler

    def get_train_dataloader(self):
        if not self.train_dataset or not self.use_sampler:
            return super().get_train_dataloader()

        # Create WeightedRandomSampler
        labels = np.array([int(x["labels"].item()) if isinstance(x["labels"], torch.Tensor) else int(x["labels"])
                           for x in self.train_dataset])
        class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
        weight_per_class = 1.0 / class_sample_count
        samples_weight = np.array([weight_per_class[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight).double()
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels").to(model.device)
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss



# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    seed=42
)

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = (labels == preds).mean()
    return {"accuracy": acc, "precision": float(p), "recall": float(r), "f1": float(f1)}

#enable sampler if dataset is still imbalanced
use_sampler = True
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    use_sampler=use_sampler
)

# Train
trainer.train()

# Evaluate on test set
print("\n TEST METRICS ")
test_metrics = trainer.evaluate(eval_dataset=test_dataset)
print(test_metrics)

# Save
trainer.save_model("saved_model")
tokenizer.save_pretrained("saved_model")

# Prediction helper
def predict(text, thresh=0.5):
    model.eval()
    text = clean_text(text)
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    device = trainer.model.device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=1).item()
        confidence = float(probs[0][pred].cpu().numpy())
    return {"label": int(pred), "prob": confidence, "text": text}

# quick check
print("\n= SAMPLE TEST ==")
print(predict("I hate you"))
print(predict("म तिमीलाई मन पराउँदिन", ))
