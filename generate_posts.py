#!/usr/bin/env python3
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- LOGGING (FIXED: Open file once, reuse handle) ----
log_file = "logs.txt"
log_handle = open(log_file, "a", encoding="utf-8")

def log(msg):
    print(msg)
    log_handle.write(msg + "\n")
    log_handle.flush()  # Live update in GitHub logs

log("AI Generation Started - Fixed for CPU in GitHub Actions")

# ---- PAYLOAD ----
site_url = os.getenv("SITE_URL", "")
topics_str = os.getenv("TOPICS", "")
site_desc = os.getenv("SITE_DESC", "a general blog")

topics = [t.strip() for t in topics_str.split(",") if t.strip()][:15]
if len(topics) < 15:
    topics += ["General Tips"] * (15 - len(topics))

log(f"Site: {site_url}")
log(f"Topics: {topics}")
log(f"Site Description: {site_desc}")

# ---- MODEL (CPU-safe: float32 + no device_map) ----
log("Loading Mistral-7B-Instruct-v0.3 (CPU mode)...")
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,      # CPU-safe
        device_map=None,                # No sharding
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()
    log("Model loaded successfully (float32 on CPU)")
except Exception as e:
    log(f"Model load failed: {e}")
    log("Falling back to smaller model...")
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    log(f"Fallback model loaded: {model_name}")

def generate_article(title):
    log(f"Generating: {title}")
    prompt = f"<s>[INST] Write a detailed blog post (400-600 words) about '{title}' for {site_desc}. Include intro, 3-5 tips, examples, and conclusion. Natural tone. [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Clean up response
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1].strip()
    word_count = len(text.split())
    log(f"Done: {title} ({word_count} words)")
    return text

# ---- GENERATE ----
articles = []
progress = {"total": len(topics), "done": 0, "current": "", "percent": 0}

log("Starting generation loop...")

for i, title in enumerate(topics, 1):
    progress["current"] = title
    progress["done"] = i - 1
    progress["percent"] = int((i - 1) / len(topics) * 100)
    
    # Save progress
    with open("progress.json", "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)
    log(f"Progress: {progress['percent']}%")

    # Generate article
    content = generate_article(title)
    articles.append({"title": title, "content": content})

    # Update progress
    progress["done"] = i
    progress["percent"] = int(i / len(topics) * 100)
    with open("progress.json", "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)

# ---- SAVE FINAL FILES ----
# Save articles
with open("articles.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, indent=2, ensure_ascii=False)
log("articles.json saved")

# Final progress
progress["percent"] = 100
with open("progress.json", "w", encoding="utf-8") as f:
    json.dump(progress, f, ensure_ascii=False)
log("progress.json updated to 100%")

log("All 15 AI posts generated!")

# Close log file
log_handle.close()
print("SUCCESS: Generation complete.")
