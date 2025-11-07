#!/usr/bin/env python3
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- LOGGING (open once, close at end) ----------
log_file = "logs.txt"
log_handle = open(log_file, "a", encoding="utf-8")

def log(msg):
    print(msg)
    log_handle.write(msg + "\n")
    log_handle.flush()

log("AI Generation Started – GPT-2 Medium (CPU-safe)")

# ---------- INPUT ----------
site_url   = os.getenv("SITE_URL", "")
topics_str = os.getenv("TOPICS", "")
site_desc  = os.getenv("SITE_DESC", "a general blog")

topics = [t.strip() for t in topics_str.split(",") if t.strip()][:15]
if len(topics) < 15:
    topics += ["General Tips"] * (15 - len(topics))

log(f"Site: {site_url}")
log(f"Site description: {site_desc}")
log(f"Topics ({len(topics)}): {topics}")

# ---------- MODEL ----------
log("Loading gpt2-medium ...")
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Pad token required for generation
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,   # CPU-safe
    low_cpu_mem_usage=True
)
model.eval()
log("Model loaded successfully")

def generate_article(title: str) -> str:
    log(f"Generating: {title}")
    prompt = (
        f"Write a detailed blog post (400-600 words) about \"{title}\" "
        f"for {site_desc}. Include an intro, 3-5 practical tips, "
        f"real-world examples, and a conclusion. Use a friendly tone.\n\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.8,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Remove the prompt part
    article = text[len(prompt):].strip()
    word_count = len(article.split())
    log(f"Done: {title} ({word_count} words)")
    return article

# ---------- GENERATION LOOP ----------
articles = []
progress = {"total": len(topics), "done": 0, "current": "", "percent": 0}

log("Starting generation loop...")

for i, title in enumerate(topics, 1):
    # ---- update progress before generation ----
    progress["current"] = title
    progress["done"]    = i - 1
    progress["percent"] = int((i - 1) / len(topics) * 100)
    with open("progress.json", "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)

    # ---- generate ----
    content = generate_article(title)
    articles.append({"title": title, "content": content})

    # ---- update progress after generation ----
    progress["done"]    = i
    progress["percent"] = int(i / len(topics) * 100)
    with open("progress.json", "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)

# ---------- FINAL SAVE ----------
with open("articles.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, indent=2, ensure_ascii=False)
log("articles.json saved")

progress["percent"] = 100
with open("progress.json", "w", encoding="utf-8") as f:
    json.dump(progress, f, ensure_ascii=False)
log("progress.json set to 100%")

log("All 15 AI posts generated successfully!")
log_handle.close()
print("SUCCESS")
