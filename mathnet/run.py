#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 14:26:03 2025

@author: abheekpathirana
"""

import re
import ast
import operator
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification
import json
from datetime import datetime

# Generation parameters
GEN_TEMPERATURE = 0.7   # creativity/randomness
GEN_TOP_K = 50          # diversity limit by top-k
GEN_TOP_P = 0.9         # diversity limit by nucleus sampling

# Gating confidence threshold
GATING_CONF_THRESHOLD = 0.5 # between 0 and 1

# Paths to your saved models/tokenizers
GATING_MODEL_PATH = "MathNet/gating_gpt2_model/best_model"
GATING_TOKENIZER_PATH = "MathNet/gating_gpt2_model/best_tokenizer"

ALG_MODEL_PATH = "MathNet/algebraic_gpt2/best"
ALG_TOKENIZER_PATH = "MathNet/algebraic_gpt2/best"

ARITH_MODEL_PATH = "MathNet/arithmetic_gpt2/best"
ARITH_TOKENIZER_PATH = "MathNet/arithmetic_gpt2/best"

DEVICE = torch.device("cpu")



# Load gating model + tokenizer (classification)
gating_tokenizer = GPT2Tokenizer.from_pretrained(GATING_TOKENIZER_PATH)
gating_model = GPT2ForSequenceClassification.from_pretrained(GATING_MODEL_PATH)
gating_model.to(DEVICE)
gating_model.eval()

# Load algebraic and arithmetic models + tokenizers (generation)
alg_tokenizer = GPT2Tokenizer.from_pretrained(ALG_TOKENIZER_PATH)
alg_model = GPT2LMHeadModel.from_pretrained(ALG_MODEL_PATH)
alg_model.to(DEVICE)
alg_model.eval()

arith_tokenizer = GPT2Tokenizer.from_pretrained(ARITH_TOKENIZER_PATH)
arith_model = GPT2LMHeadModel.from_pretrained(ARITH_MODEL_PATH)
arith_model.to(DEVICE)
arith_model.eval()




print(" ")
print("Welcome to MathNet")
print(" ")
print("The collection of Worlds smallest tool calling SLMs")
print(" ")
print(" ")
print("MathNet is a system consisting of 3 different Small Language Models working in sync to perform basic arithmetic and basic algebraic tasks.")
print(" ")
print(" ")
print("Model Parameters: down below")
print(" ")
print(f"Algebraic model parameters: {sum(p.numel() for p in alg_model.parameters())}")
print(f"Arithmetic model parameters: {sum(p.numel() for p in arith_model.parameters())}")
print(f"Gating model parameters: {sum(p.numel() for p in gating_model.parameters())}")
print(" ")
print(" ")
print("Task oriented pretraining was done using purely synthetic datasets which exceed 6million tokens.")
print(" ")
print(f" Total number of parameters = {sum(p.numel() for p in alg_model.parameters())+sum(p.numel() for p in arith_model.parameters())+sum(p.numel() for p in gating_model.parameters())}.")
print("                             (14,231,520)")
print(" ")
print("The 2 experts present in the system is able to use tools such as calculators for their respective tasks.")
print(" ")
print("started progress in 2/7/2025 and completed in 15/7/2025.")
print(" ")
print("A property of abheekpathirana.")
print(" ")
print("More info at - https://github.com/abheek-pathirana/MathNet/tree/main")
print(" ")
print(f" MathNet is set to {DEVICE} on your device.\n")


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
        if isinstance(node, ast.Num):  # Python <3.8
            return node.n
        elif isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.BinOp):
            return allowed_operators[type(node.op)](eval_node(node.left), eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            return allowed_operators[type(node.op)](eval_node(node.operand))
        else:
            raise TypeError("Unsupported expression")

    tree = ast.parse(expr, mode='eval')
    return eval_node(tree.body)



def extract_and_eval(output):
    match = re.search(r"<math_call_start>(.*?)<math_call_end>", output)
    if not match:
        return "No math expression found."
    expr = match.group(1).strip()
    try:
        result = safe_eval(expr)
        return f"Expression: {expr} = {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"
    
    
    
####################3
"""
def classify_gating(text):
    inputs = gating_tokenizer(text, return_tensors="pt", truncation=True, max_length=64, padding="max_length")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = gating_model(**inputs)
    logits = outputs.logits
    pred_label = torch.argmax(logits, dim=-1).item()
    print(f"Gating Model logits: {logits}")
    print(f"Gating Model predicted label: {pred_label}")
    return pred_label
"""
import torch.nn.functional as F

def classify_gating(text):
    inputs = gating_tokenizer(text, return_tensors="pt", truncation=True, max_length=64, padding="max_length")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = gating_model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    conf, pred_label = torch.max(probs, dim=-1)
    conf = conf.item()
    pred_label = pred_label.item()

    print(f"Gating Model probabilities: {probs}")
    print(" ")
    print(f"Predicted label: {pred_label} with confidence: {conf:.4f}")

    # Apply threshold — fallback to arithmetic if not confident
    if conf < GATING_CONF_THRESHOLD:
        print("Gating model not confident — defaulting to Arithmetic model")
        return 0
    return pred_label


def generate_text(model, tokenizer, text, max_length=64, temperature=GEN_TEMPERATURE, top_k=GEN_TOP_K, top_p=GEN_TOP_P):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text



def full_pipeline(text):
    gating_label = classify_gating(text)

    # Wrap text with <start_prompt> and <end_prompt> for routed models
    wrapped_text = f"<start_prompt>{text}<end_prompt>"

    if gating_label == 1:
        print(" ")
        print("Routing to Algebraic model")
        print(" ")
        output_text = generate_text(alg_model, alg_tokenizer, wrapped_text)
    else:
        print(" ")
        print("Routing to Arithmetic model")
        print(" ")
        output_text = generate_text(arith_model, arith_tokenizer, wrapped_text)
    return output_text

FEEDBACK_FILE = "feedback.jsonl"

def save_feedback(prompt, response, feedback):
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt,
        "response": response,
        "feedback": feedback  # "good" or "bad"
    }
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")



if __name__ == "__main__":
    while True:

        inp = input("Enter math question (or 'exit'): ").strip()
        if not inp:  # If string is empty after stripping
            print("Please enter a valid math question.")
            continue
        
        if inp.lower() == "exit":
            print("Goodbye 👋")
            break
        
        if not re.search(r"[0-9+\-*/=^()]", inp):
            print("Sorry Iam only able to reply to basic math related queries.")
            continue
        
            break
        output = full_pipeline(inp)
        print("Final output from routed model:")
        print(" ")
        print(output)
        print(" ")
     
        print(f" {extract_and_eval(output)}\n")

        feedback = input("Was this response good or bad? enter either (g/b) or press enter to skip: ").strip().lower()
        if feedback in ["g", "good"]:
            save_feedback(inp, output, "good")
        elif feedback in ["b", "bad"]:
            save_feedback(inp, output, "bad")
        else:
            print("Feedback skipped.")

