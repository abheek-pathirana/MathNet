#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MathNet Benchmark on ProCreations/SimpleMath with batching and logging
"""

import re
import ast
import operator
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F

# ---------------------- Settings ----------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
GEN_TEMPERATURE = 0.1
GEN_TOP_K = 10
GEN_TOP_P = 0.1
GATING_CONF_THRESHOLD = 0.5

# ---------------------- Model Paths ----------------------
GATING_MODEL_PATH = "MathNet/gating_gpt2_model/best_model"
GATING_TOKENIZER_PATH = "MathNet/gating_gpt2_model/best_tokenizer"

ALG_MODEL_PATH = "MathNet/algebraic_gpt2/best"
ALG_TOKENIZER_PATH = "MathNet/algebraic_gpt2/best"

ARITH_MODEL_PATH = "MathNet/arithmetic_gpt2/best"
ARITH_TOKENIZER_PATH = "MathNet/arithmetic_gpt2/best"

# ---------------------- Load Models ----------------------
gating_tokenizer = GPT2Tokenizer.from_pretrained(GATING_TOKENIZER_PATH)
gating_model = GPT2ForSequenceClassification.from_pretrained(GATING_MODEL_PATH).to(DEVICE)
gating_model.eval()

alg_tokenizer = GPT2Tokenizer.from_pretrained(ALG_TOKENIZER_PATH)
alg_model = GPT2LMHeadModel.from_pretrained(ALG_MODEL_PATH).to(DEVICE)
alg_model.eval()

arith_tokenizer = GPT2Tokenizer.from_pretrained(ARITH_TOKENIZER_PATH)
arith_model = GPT2LMHeadModel.from_pretrained(ARITH_MODEL_PATH).to(DEVICE)
arith_model.eval()

# ---------------------- Safe Evaluation ----------------------
def safe_eval(expr):
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg
    }
    def eval_node(node):
        if isinstance(node, ast.Num): return node.n
        elif isinstance(node, ast.Constant): return node.value
        elif isinstance(node, ast.BinOp): return allowed_operators[type(node.op)](eval_node(node.left), eval_node(node.right))
        elif isinstance(node, ast.UnaryOp): return allowed_operators[type(node.op)](eval_node(node.operand))
        else: raise TypeError("Unsupported expression")
    tree = ast.parse(expr, mode='eval')
    return eval_node(tree.body)

def extract_and_eval(output):
    match = re.search(r"<math_call_start>(.*?)<math_call_end>", output)
    if not match: return None
    expr = match.group(1).strip()
    try:
        return str(safe_eval(expr))
    except:
        return None

# ---------------------- Gating ----------------------
def classify_gating(text):
    inputs = gating_tokenizer(text, return_tensors="pt", truncation=True, max_length=64, padding="max_length").to(DEVICE)
    with torch.no_grad():
        logits = gating_model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    conf, pred_label = torch.max(probs, dim=-1)
    conf = conf.item()
    pred_label = pred_label.item()
    if conf < GATING_CONF_THRESHOLD:  # fallback to arithmetic
        return 0
    return pred_label

# ---------------------- Generation ----------------------
def generate_text(model, tokenizer, text, max_length=64):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def full_pipeline(text):
    label = classify_gating(text)
    wrapped_text = f"<start_prompt>{text}<end_prompt>"
    if label == 1:
        output = generate_text(alg_model, alg_tokenizer, wrapped_text)
    else:
        output = generate_text(arith_model, arith_tokenizer, wrapped_text)
    return output

# ---------------------- Benchmark Loop ----------------------
if __name__ == "__main__":
    dataset = load_dataset("ProCreations/SimpleMath")["train"]
    correct = 0
    total = len(dataset)

    for idx, row in enumerate(tqdm(dataset, total=total)):
        problem = row["problem"]
        gold_answer = str(row["answer"]).strip()

        output = full_pipeline(problem)
        pred_answer = extract_and_eval(output)

        if pred_answer == gold_answer:
            correct += 1

        # Log every 100 examples
        if (idx + 1) % 100 == 0:
            print(f"[{idx + 1}/{total}] Problem: {problem}")
            print(f"Expected: {gold_answer}, Predicted: {pred_answer}")
            print(f"Current Accuracy: {correct / (idx + 1):.4%}\n")

    accuracy = correct / total
    print(f"\n--- Benchmark Results ---")
    print(f"Total examples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4%}")