import os
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import NewPost

print("Loading payload...")
payload = json.loads(os.getenv('GITHUB_EVENT_PAYLOAD', '{}'))
site_url = payload.get('site_url', '').rstrip('/')
wp_user = payload.get('wp_user', '')
wp_pass = payload.get('wp_pass', '')
topics = [t.strip() for t in payload.get('topics', '').split(',') if t.strip()][:15]

if len(topics) < 15:
    topics += ["General Blogging Tips"] * (15 - len(topics))

print(f"Generating {len(topics)} posts for: {site_url}")

# Fix legacy warning
print("Loading FLAN-T5-Large (legacy=False)...")
device = torch.device("cpu")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)
model.eval()
print("Model loaded.")

def generate_article(title):
    print(f"  Generating: {title}")
    prompt = f"Write a detailed blog post (400-600 words) about: {title}. Include intro, 3-5 tips, examples, and conclusion. Natural tone."
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=600,
        num_beams=5,
        temperature=0.7,
        do_sample=True,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Done: {len(text.split())} words")
    return text

# Connect
wp = Client(f"{site_url}/xmlrpc.php", wp_user, wp_pass)

articles = []

for i, title in enumerate(topics, 1):
    print(f"[{i}/15] Processing: {title}")
    content = generate_article(title)
    articles.append({
        "title": title,
        "content": content,
        "status": "draft"
    })

# Save to JSON in repo
json_path = "/tmp/articles.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(articles, f, indent=2, ensure_ascii=False)
print(f"Saved {len(articles)} articles to {json_path}")

# Upload JSON as artifact
print("Uploading articles.json as artifact...")
import shutil
shutil.copy(json_path, "articles.json")
print("JSON ready for download.")
