#!/usr/bin/env python3
import os
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load payload
payload = json.loads(os.getenv("GITHUB_EVENT_PAYLOAD", "{}"))
site_url = payload.get("site_url", "").rstrip("/")
topics = [t.strip() for t in payload.get("topics", "").split(",") if t.strip()][:15]

if len(topics) < 15:
    topics += ["General Tips"] * (15 - len(topics))

print(f"Generating {len(topics)} posts for {site_url}")

# Load model
print("Loading google/flan-t5-large...")
device = torch.device("cpu")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)
model.eval()
print("Model loaded")

def generate_article(title):
    print(f"  → {title}")
    prompt = f"Write a detailed blog post (400-600 words) about: {title}. Include intro, tips, examples, conclusion."
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(**inputs, max_length=600, num_beams=5, temperature=0.7, do_sample=True, no_repeat_ngram_size=2)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  ← {len(text.split())} words")
    return text

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

with open("articles.json", "w") as f:
    json.dump(articles, f, indent=2)

print("All 15 AI posts generated!")
