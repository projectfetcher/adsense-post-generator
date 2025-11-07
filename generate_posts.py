#!/usr/bin/env python3
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === LOGGING ===
log_file = "logs.txt"
with open(log_file, "w") as f:
    f.write("AI Generation Started - Using Mistral-7B-Instruct\n")

def log(msg):
    print(msg)
    with open(log_file, "a") as f:
        f.write(f"{msg}\n")
    # Flush to make visible immediately
    f.flush()

# === PAYLOAD ===
payload = json.loads(os.getenv("GITHUB_EVENT_PAYLOAD", "{}"))
site_url = payload.get("site_url", "").rstrip("/")
wp_user = payload.get("wp_user", "")
wp_pass = payload.get("wp_pass", "")
topics = [t.strip() for t in payload.get("topics", "").split(",") if t.strip()][:15]
site_description = payload.get("site_description", "a general blog")

if len(topics) < 15:
    topics += ["General Tips"] * (15 - len(topics))

log(f"Site: {site_url}")
log(f"Description: {site_description}")
log(f"Topics: {topics}")

# === MODEL (Mistral-7B – powerful for articles) ===
log("Loading Mistral-7B-Instruct-v0.3 (CPU)...")
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu")
model.eval()
log("Model loaded")

def generate_article(title):
    log(f"Generating: {title}")
    prompt = f"<s>[INST] Write a detailed blog post (400-600 words) about '{title}' for {site_description}. Include intro, 3-5 tips, examples, and conclusion. Natural tone. [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(
        **inputs,
        max_new_tokens=600,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
    word_count = len(text.split())
    log(f"Done: {title} ({word_count} words)")
    return text

# === GENERATE ===
articles = []
progress = {"total": len(topics), "done": 0, "current": "", "percent": 0}

for i, title in enumerate(topics, 1):
    progress["current"] = title
    progress["done"] = i - 1
    progress["percent"] = int((i - 1) / len(topics) * 100)
    with open("progress.json", "w") as f:
        json.dump(progress, f)

    content = generate_article(title)
    articles.append({"title": title, "content": content})

    progress["done"] = i
    progress["percent"] = int(i / len(topics) * 100)
    with open("progress.json", "w") as f:
        json.dump(progress, f)

# === SAVE ===
with open("articles.json", "w") as f:
    json.dump(articles, f, indent=2)
log("All 15 AI posts generated!")
