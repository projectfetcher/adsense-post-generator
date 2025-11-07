#!/usr/bin/env python3
import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- LOGGING ----
log_file = "logs.txt"
def log(msg):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f: f.write(msg + "\n")

# ---- PAYLOAD ----
payload = {
    "site_url": os.getenv("SITE_URL",""),
    "wp_user":  os.getenv("WP_USER",""),
    "wp_pass":  os.getenv("WP_PASS",""),
    "topics":   os.getenv("TOPICS",""),
    "site_description": os.getenv("SITE_DESC","a general blog")
}
log(f"Payload received – site: {payload['site_url']}")

topics = [t.strip() for t in payload["topics"].split(",") if t.strip()][:15]
if len(topics)<15: topics += ["General Tips"]*(15-len(topics))
log(f"Topics: {topics}")

# ---- MODEL ----
log("Loading Mistral-7B-Instruct-v0.3 …")
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu")
model.eval()
log("Model ready")

def generate(title):
    log(f"Generating: {title}")
    prompt = f"<s>[INST] Write a detailed blog post (400‑600 words) about \"{title}\" for {payload['site_description']}. Include intro, 3‑5 tips, examples, conclusion. [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=650, temperature=0.7, do_sample=True)
    text = tokenizer.decode(out[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
    log(f"Done: {title} ({len(text.split())} words)")
    return text

# ---- GENERATE ----
articles = []
progress = {"total":len(topics),"done":0,"current":"","percent":0}

for i, title in enumerate(topics,1):
    progress.update({"current":title,"done":i-1,"percent":int((i-1)/len(topics)*100)})
    with open("progress.json","w") as f: json.dump(progress,f)

    articles.append({"title":title,"content":generate(title)})

    progress.update({"done":i,"percent":int(i/len(topics)*100)})
    with open("progress.json","w") as f: json.dump(progress,f)

with open("articles.json","w") as f: json.dump(articles,f,indent=2)
log("All 15 AI posts generated!")
