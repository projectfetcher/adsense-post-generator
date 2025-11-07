#!/usr/bin/env python3
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# ---------- LOGGING ----------
log_file = "logs.txt"
log_handle = open(log_file, "a", encoding="utf-8")

def log(msg):
    print(msg)
    log_handle.write(msg + "\n")
    log_handle.flush()

log("AI Generation Started – Flan-T5-Large + MiniLM (CPU)")

# ---------- INPUT ----------
site_url   = os.getenv("SITE_URL", "")
topics_str = os.getenv("TOPICS", "")
site_desc  = os.getenv("SITE_DESC", "a general blog")

topics = [t.strip() for t in topics_str.split(",") if t.strip()][:15]
if len(topics) < 15:
    topics += ["General Tips"] * (15 - len(topics))

log(f"Site: {site_url}")
log(f"Description: {site_desc}")
log(f"Topics ({len(topics)}): {topics}")

# ---------- MODEL & TOKENIZER ----------
log("Loading google/flan-t5-large ...")
device = torch.device("cpu")  # Always CPU
model_name = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()
model.to(device)
log("Flan-T5-Large loaded on CPU")

# ---------- SENTENCE TRANSFORMER ----------
log("Loading all-MiniLM-L6-v2 for similarity...")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
log("SentenceTransformer loaded")

# ---------- GENERATION ----------
def generate_article(title: str) -> str:
    log(f"Generating: {title}")
    prompt = (
        f"Write a detailed blog post (400-600 words) about \"{title}\" "
        f"for {site_desc}. Include an intro, 3-5 practical tips, "
        f"real-world examples, and a conclusion. Use a friendly tone."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )

    article = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    word_count = len(article.split())
    log(f"Done: {title} ({word_count} words)")

    # Optional: Use similarity to deduplicate (skip if too similar to previous)
    if articles:
        prev = articles[-1]["content"]
        emb1 = similarity_model.encode(prev, convert_to_tensor=True)
        emb2 = similarity_model.encode(article, convert_to_tensor=True)
        sim = torch.cosine_similarity(emb1, emb2, dim=0).item()
        log(f"Similarity to previous: {sim:.3f}")
        if sim > 0.85:
            log("Too similar – regenerating...")
            return generate_article(title)  # retry

    return article

# ---------- LOOP ----------
articles = []
progress = {"total": len(topics), "done": 0, "current": "", "percent": 0}

log("Starting generation loop...")

for i, title in enumerate(topics, 1):
    # Update progress
    progress["current"] = title
    progress["done"] = i - 1
    progress["percent"] = int((i - 1) / len(topics) * 100)
    with open("progress.json", "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)

    # Generate
    content = generate_article(title)
    articles.append({"title": title, "content": content})

    # Final progress
    progress["done"] = i
    progress["percent"] = int(i / len(topics) * 100)
    with open("progress.json", "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)

# ---------- SAVE ----------
with open("articles.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, indent=2, ensure_ascii=False)
log("articles.json saved")

progress["percent"] = 100
with open("progress.json", "w", encoding="utf-8") as f:
    json.dump(progress, f, ensure_ascii=False)
log("progress.json set to 100%")

log("All 15 AI posts generated!")
log_handle.close()
print("SUCCESS")
