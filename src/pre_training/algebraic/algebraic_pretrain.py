#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 18:50:29 2025

@author: abheekpathirana
"""

from transformers import AutoTokenizer
import torch


# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

# === Load raw dataset ===
with open("SML_pre-train.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# === Tokenize ===
tokens = tokenizer.encode(raw_text, add_special_tokens=False)

# === Chunk into 256-token sequences ===
chunk_size = 256
chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]

# === Pad final chunk if shorter ===
last_chunk = chunks[-1]
if len(last_chunk) < chunk_size:
    chunks[-1] = last_chunk + [tokenizer.pad_token_id] * (chunk_size - len(last_chunk))

# === Convert to tensor ===
tensor_data = torch.tensor(chunks)

# === Save as torch file (optional) ===
torch.save(tensor_data, "chunked_data1(new).pt")

print(f"✅ Done! Total chunks: {len(chunks)} | Shape: {tensor_data.shape}")
from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config(
    vocab_size=50257,   # GPT-2 tokenizer vocab
    n_positions=256,     # context#64 for the arithmetic model
    n_ctx=256,           #64(if you are tuning this hyperparameter then make sure both this and "n_positions" have the same number) and make sure u edit the "chunk_size=256" to the same value as this.
    n_embd=96,          # Hidden size
    n_layer=7,          # Layers (transformer blocks) #7 for algebraic and 4 for arithmetic (4 for algebraic would be fine but i trained it on 7)
    n_head=4,           # Attention heads
)

def generate_sample_text(model, tokenizer, prompt="the man", max_length=50):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    model.train()
    return generated_text

model = GPT2LMHeadModel(config)
print(model)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,} | Trainable: {trainable:,}")

import torch
from torch.utils.data import Dataset, DataLoader

class TokenChunkDataset(Dataset):
    def __init__(self, tensor_data):
        self.data = tensor_data

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        return {'input_ids': x, 'labels': x}  # Causal LM: input == target

# Load your tensor (from last step)
chunks = torch.load("chunked_data1(new).pt")  # If you saved with torch.save(...)
dataset = TokenChunkDataset(chunks)
loader = DataLoader(dataset, batch_size=35, shuffle=True)

from transformers import AdamW
from tqdm import tqdm

optimizer = AdamW(model.parameters(), lr=1e-4)
device = torch.device("mps" if torch.mps.is_available() else "cpu")#apple gpu or cpu 
print(device)
model = model.to(device)
model.train()

epochs = 70 #this is nearly 18 hours on a apple m3 macbook air! so choose wisely the number of epochs you go for, id reccomend sticking to 20 epochs since this is only the base of the model
for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    pbar = tqdm(loader)
    for batch in pbar:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description(f"Loss: {loss.item():.4f}")
    print("\nSample generation after epoch", epoch+1)
    print(generate_sample_text(model, tokenizer))
    model.save_pretrained(f"SML_algebraic_pt{epoch+1}")#add any name you want


    tokenizer.save_pretrained(f"SML_algebraic_pt{epoch+1}")#add any name you want


print("\nSample generation after epoch", epoch+1)
print(generate_sample_text(model, tokenizer))
        
model.save_pretrained("MathNet_pretrain_m2")


tokenizer.save_pretrained("MathNet_pretrain_t2")



