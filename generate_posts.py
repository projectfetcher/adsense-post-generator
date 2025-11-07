#!/usr/bin/env python3
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Logging
log_file = "logs.txt"
def log(msg):
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

# Payload
site_url = os.getenv("SITE_URL", "")
wp_user = os.getenv("WP_USER", "")
wp_pass = os.getenv("WP_PASS", "")
topics_str = os.getenv("TOPICS", "")
site_desc = os.getenv("SITE_DESC", "a general blog")

topics = [t.strip() for t in topics_str.split(",") if t.strip()][:15]
if len(topics) < 15:
    topics += ["General Tips"] * (15 - len(topics))

log(f"Site: {site_url}")
log(f"Topics: {topics}")

# Model
log("Loading Mistral-7B...")
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu")
model.eval()
log("Model loaded")

def generate(title):
    log(f"Generating: {title}")
    prompt = f"<s>[INST] Write a detailed blog post (400-600 words) about '{title}' for {site_desc}. Include intro, 3-5 tips, examples, conclusion. [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=600, temperature=0.7, do_sample=True)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
    log(f"Done: {title} ({len(text.split())} words)")
    return text

# Generate
articles = []
progress = {"total": len(topics), "done": 0, "current": "", "percent": 0}

for i, title in enumerate(topics, 1):
    progress["current"] = title
    progress["done"] = i - 1
    progress["percent"] = int((i - 1) / len(topics) * 100)
    with open("progress.json", "w") as f:
        json.dump(progress, f)

    content = generate(title)
    articles.append({"title": title, "content": content})

    progress["done"] = i
    progress["percent"] = int(i / len(topics) * 100)
    with open("progress.json", "w") as f:
        json.dump(progress, f)

with open("articles.json", "w") as f:
    json.dump(articles, f, indent=2)
log("All 15 AI posts generated!")
