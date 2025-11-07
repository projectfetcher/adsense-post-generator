#!/usr/bin/env python3
import os
import json
import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import NewPost

# -------------------------------------------------
# 1. Load payload
# -------------------------------------------------
print("Loading payload...")
payload = json.loads(os.getenv("GITHUB_EVENT_PAYLOAD", "{}"))
site_url = payload.get("site_url", "").rstrip("/")
wp_user  = payload.get("wp_user", "")
wp_pass  = payload.get("wp_pass", "")
topics   = [t.strip() for t in payload.get("topics", "").split(",") if t.strip()][:15]

if len(topics) < 15:
    topics += ["General Blogging Tips"] * (15 - len(topics))

print(f"Generating {len(topics)} posts for {site_url}")

# -------------------------------------------------
# 2. Install missing protobuf (GitHub runners miss it)
# -------------------------------------------------
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "protobuf>=4.21"])

# -------------------------------------------------
# 3. Load model (legacy=False – no warning)
# -------------------------------------------------
print("Loading google/flan-t5-large (legacy=False)...")
device    = torch.device("cpu")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=False)
model     = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)
model.eval()
print("Model loaded")

# -------------------------------------------------
# 4. Helper – generate one article
# -------------------------------------------------
def generate_article(title: str) -> str:
    print(f"  → Generating: {title}")
    prompt = (
        f"Write a detailed blog post (400-600 words) about: {title}. "
        "Include intro, 3-5 practical tips, real-world examples, and a conclusion. "
        "Use a natural, engaging tone."
    )
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=600,
        num_beams=5,
        temperature=0.7,
        do_sample=True,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  ← Done ({len(text.split())} words)")
    return text

# -------------------------------------------------
# 5. Connect to WordPress
# -------------------------------------------------
wp = Client(f"{site_url}/xmlrpc.php", wp_user, wp_pass)

# -------------------------------------------------
# 6. Generate + save progress + final JSON
# -------------------------------------------------
articles   = []
progress   = {"total": len(topics), "done": 0, "current": "", "percent": 0}
progress_path = "/tmp/progress.json"
final_path    = "/tmp/articles.json"

for idx, title in enumerate(topics, 1):
    # ---- update progress (per article) ----
    progress["current"] = title
    progress["done"]    = idx - 1
    progress["percent"] = int((idx - 1) / len(topics) * 100)
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f)
    # copy to repo root so WP can read it
    os.system(f"cp {progress_path} progress.json")

    # ---- generate article ----
    content = generate_article(title)
    articles.append({"title": title, "content": content})

    # ---- update progress again (article finished) ----
    progress["done"]    = idx
    progress["percent"] = int(idx / len(topics) * 100)
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f)
    os.system(f"cp {progress_path} progress.json")

# -------------------------------------------------
# 7. Save final articles.json (artifact)
# -------------------------------------------------
with open(final_path, "w", encoding="utf-8") as f:
    json.dump(articles, f, indent=2, ensure_ascii=False)
os.system("cp /tmp/articles.json articles.json")
print("All 15 AI posts generated → articles.json + progress.json")

# -------------------------------------------------
# 8. (Optional) Publish directly – comment out if you only want JSON
# -------------------------------------------------
# for a in articles:
#     post = WordPressPost()
#     post.title = a["title"]
#     post.content = f"<h2>{a['title']}</h2><p>{a['content']}</p>"
#     post.terms_names = {"category": ["Blog"]}
#     post.post_status = "publish"
#     wp.call(NewPost(post))
# print("All posts published to WordPress")
