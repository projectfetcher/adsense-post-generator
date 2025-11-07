import os
import json
from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import NewPost
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load payload
payload = json.loads(os.getenv('GITHUB_EVENT_PAYLOAD', '{}'))
site_url = payload.get('site_url', '').rstrip('/')
wp_user = payload.get('wp_user', '')
wp_pass = payload.get('wp_pass', '')
topics_input = payload.get('topics', '')

if not all([site_url, wp_user, wp_pass, topics_input]):
    print("Missing data")
    exit(1)

# Parse topics
topics = [t.strip() for t in topics_input.split(',') if t.strip()][:15]
if len(topics) < 15:
    topics.extend(["General Blogging Tips"] * (15 - len(topics)))

# Load AI model
print("Loading google/flan-t5-large (CPU)...")
device = torch.device("cpu")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(device)
model.eval()

def generate_article(title):
    prompt = f"Write a detailed blog post (400-600 words) about: {title}. Include intro, 3-5 tips, examples, and conclusion. Natural tone."
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=600,
        num_beams=5,
        temperature=0.7,
        do_sample=True,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Connect & publish
wp = Client(f"{site_url}/xmlrpc.php", wp_user, wp_pass)

for i, title in enumerate(topics, 1):
    print(f"Generating {i}/15: {title}")
    content = generate_article(title)
    post = WordPressPost()
    post.title = title
    post.content = f"<h2>{title}</h2><p>{content}</p>"
    post.terms_names = {'category': ['Blog', 'Tips']}
    post.post_status = 'publish'
    wp.call(NewPost(post))
    print(f"Published: {title}")

print("All 15 AI posts published!")
