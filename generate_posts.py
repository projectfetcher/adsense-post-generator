#!/usr/bin/env python3
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- LOGGING ----
log_file = "logs.txt"
def log(msg):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    f.flush()  # Ensure immediate write for live logs

log("AI Generation Started - Fixed for CPU in GitHub Actions")

# ---- PAYLOAD ----
payload = {
    "site_url": os.getenv("SITE_URL", ""),
    "wp_user": os.getenv("WP_USER", ""),
    "wp_pass": os.getenv("WP_PASS", ""),
    "topics": os.getenv("TOPICS", ""),
    "site_description": os.getenv("SITE_DESC", "a general blog")
}
log(f"Payload: site={payload['site_url']}, topics={payload['topics']}")

topics = [t.strip() for t in payload["topics"].split(",") if t.strip()][:15]
if len(topics) < 15:
    topics += ["General Tips"] * (15 - len(topics))

log(f"Final topics: {topics}")

# ---- MODEL (Fixed for CPU: float32 + no device_map) ----
log("Loading Mistral-7B-Instruct-v0.3 (CPU-safe mode)...")
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # FIXED: Use float32 for CPU (avoids dtype/sharding error)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Safe for CPU – no float16 conflicts
        device_map=None,  # No sharding on CPU
        low_cpu_mem_usage=True,  # Reduce memory
        trust_remote_code=True
    )
    model.eval()
    log("Model loaded successfully (float32 on CPU)")
except Exception as e:
    log(f"Model load error: {e}")
    # Fallback to smaller model if Mistral fails
    model_name = "microsoft/DialoGPT-medium"  # Lightweight fallback
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    log(f"Fallback to {model_name}")

def generate_article(title):
    log(f"  → Generating: {title}")
    prompt = f"<s>[INST] Write a detailed blog post (400-600 words) about '{title}' for {payload['site_description']}. Include intro, 3-5 tips, examples, and conclusion. Natural, engaging tone. [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():  # Save memory
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1  # Avoid repetition
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
    word_count = len(text.split())
    log(f"  ← Done: {title} ({word_count} words)")
    return text

# ---- GENERATE ----
articles = []
progress = {"total": len(topics), "done": 0, "current": "", "percent": 0}

log("Starting article generation loop...")
for i, title in enumerate(topics, 1):
    progress["current"] = title
    progress["done"] = i - 1
    progress["percent"] = int((i - 1) / len(topics) * 100)
    with open("progress.json", "w") as f:
        json.dump(progress, f)
    log(f"Progress updated: {progress['percent']}%")

    content = generate_article(title)
    articles.append({"title": title, "content": content})

    progress["done"] = i
    progress["percent"] = int(i / len(topics) * 100)
    with open("progress.json", "w") as f:
        json.dump(progress, f)

log("All articles generated!")

# ---- SAVE ----
with open("articles.json", "w", encoding="utf-3") as f:
    json.dump(articles, f, indent=2, ensure_ascii=False)
log("Files saved: progress.json, articles.json, logs.txt")

print("SUCCESS: Generation complete.")
