#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 21:25:24 2025

@author: abheekpathirana
"""


from transformers import AutoTokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import os

# === Paths ===
tokenizer_path = "MathNet/src/pre_training/algebraic/SML_arithmetic_pt70" #the path might differ and the last to digits in the path may differ depending on the number of epochs trained
model_path = tokenizer_path  # Use the folder for from_pretrained()

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

# === Load and tokenize dataset ===
with open("MathNet/src/task-oriented pre-training/arithmetic/arithmetic_task_oriented_pt.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokens = tokenizer.encode(raw_text, add_special_tokens=False)

chunk_size = 64
chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
if len(chunks[-1]) < chunk_size:
    chunks[-1] += [tokenizer.pad_token_id] * (chunk_size - len(chunks[-1]))

tensor_data = torch.tensor(chunks)
torch.save(tensor_data, "chunked_data601_3(new).pt")
print(f"✅ Done! Total chunks: {len(chunks)} | Shape: {tensor_data.shape}")

# === Load model from .safetensors folder ===
print(f"🔄 Loading model from: '{model_path}'")
model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)

# === Device setup ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Sample Text Generator ===
def generate_sample_text(model, tokenizer, prompt="<start_prompt>231*3<end_prompt>", max_length=64):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_p=0.1,
            top_k=20,
            temperature=0.4,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Dataset Class ===
class TokenChunkDataset(Dataset):
    def __init__(self, tensor_data):
        self.data = tensor_data

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        return {'input_ids': x, 'labels': x}

# === Load Data ===
chunks = torch.load("chunked_data601_3(new).pt")
dataset = TokenChunkDataset(chunks)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# === Optimizer + Scheduler ===
optimizer = AdamW(model.parameters(), lr=1e-4)
epochs = 40
total_steps = len(loader) * epochs
warmup_steps = int(0.1 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# === Training Loop ===
model.train()
for epoch in range(epochs):
    print(f"\n🔁 Epoch {epoch+1}")
    pbar = tqdm(loader)
    for batch in pbar:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        pbar.set_description(f"Loss: {loss.item():.4f}")

    print("\n🧠 Sample generation after epoch", epoch + 1)
    print(generate_sample_text(model, tokenizer))

    save_dir = f"MathNet_arithmetic_v2_e{epoch+1}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

# === Final Save + Sample ===
print("\n✅ Final Sample Generation:")
print(generate_sample_text(model, tokenizer))

model.save_pretrained("arithmetic_gpt2/best")
tokenizer.save_pretrained("arithmetic_gpt2/best")