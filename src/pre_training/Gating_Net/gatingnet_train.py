#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 12:42:31 2025

@author: abheekpathirana
"""
# train_gating_gpt2.py
import os
import json
import random
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from transformers import (
    GPT2Tokenizer,
    GPT2Config,
    GPT2ForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ---------------------------
# Config / Hyperparameters
# ---------------------------
DATA_PATH = "data_shuffled556.jsonl"
OUT_DIR = "gating_gpt2_model"
BATCH_SIZE = 32
EPOCHS = 3
LR = 1e-4
MAX_LEN = 64
SEED = 42
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "mps")
PRINT_EVERY = 200

# ---------------------------
# Repro
# ---------------------------
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------
# Dataset class
# ---------------------------
class JSONLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=64):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", "").strip()
                label = int(obj.get("label", 0))
                self.samples.append((text, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",  # ensures all same length
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# ---------------------------
# Utilities
# ---------------------------
def collate_batch(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }

def compute_class_weights(dataset):
    counts = Counter()
    for _, label in dataset.samples:
        counts[label] += 1
    total = sum(counts.values())
    num_classes = len(counts)
    return torch.tensor(
        [total / (num_classes * counts.get(k, 1)) for k in range(num_classes)],
        dtype=torch.float
    )

# ---------------------------
# Main
# ---------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    print("Tokenizer vocab size:", len(tokenizer))

    # 2) Dataset
    full_dataset = JSONLDataset(DATA_PATH, tokenizer, max_len=MAX_LEN)
    n = len(full_dataset)
    print("Total samples:", n)

    # 3) Train/val split
    val_size = int(n * 0.1)
    indices = list(range(n))
    random.shuffle(indices)
    train_idx = indices[:-val_size]
    val_idx = indices[-val_size:]

    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    # 4) Class weights
    class_weights = compute_class_weights(full_dataset).to(DEVICE)
    print("Class weights:", class_weights.tolist())
    loss_fn = CrossEntropyLoss(weight=class_weights)

    # ...
    
    # 5) Model — config matches tokenizer vocab
    gate_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=MAX_LEN,
        n_ctx=MAX_LEN,
        n_embd=64,
        n_layer=2,
        n_head=4,
        num_labels=2,
    )
    
    # Initialize model once from gate_config
    model = GPT2ForSequenceClassification(gate_config)
    
    # Set pad token for model so batching works
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Resize embeddings if tokenizer was extended
    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)

# ...

    # 6) Optimizer + scheduler
    total_steps = len(train_loader) * EPOCHS
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps
    )

    # 7) Training loop
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            if step % PRINT_EVERY == 0:
                print(f"Epoch {epoch} step {step}/{len(train_loader)} avg_loss={running_loss/step:.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch} done. Train loss: {avg_train_loss:.4f} | Val acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(os.path.join(OUT_DIR, "best_model"))
            tokenizer.save_pretrained(os.path.join(OUT_DIR, "best_tokenizer"))
            print("Saved best model.")

    model.save_pretrained(os.path.join(OUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUT_DIR, "final_tokenizer"))
    print("Training complete. Best val acc:", best_val_acc)

if __name__ == "__main__":
    main()