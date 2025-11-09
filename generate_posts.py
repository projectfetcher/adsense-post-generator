#!/usr/bin/env python3
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
import random
from datetime import datetime

# ---------- CONFIG ----------
log_file = "logs.txt"
progress_file = "progress.json"
articles_file = "articles.json"
log_handle = open(log_file, "a", encoding="utf-8")

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    log_handle.write(full_msg + "\n")
    log_handle.flush()

log("AI Blog Generator Started – Using Mistral-7B-Instruct via GitHub Models (4-bit)")

# ---------- HARDCODED SITE DESCRIPTION ----------
site_desc = (
    "Mauritius.mimusjobs.com is a premier job portal dedicated to connecting talent with opportunities across Mauritius's thriving economy. "
    "From IT roles in Ebene Cybercity to luxury hospitality positions in Grand Baie, the platform features thousands of verified listings in tourism, finance, tech, healthcare, and more. "
    "Job seekers can upload resumes, build ATS-friendly profiles, and receive tailored job alerts, while employers benefit from advanced recruitment tools and company branding. "
    "With a mobile-optimized interface, multilingual support (English, French, Kreol), and AI-powered matching, it empowers locals and expatriates alike to advance their careers in one of the Indian Ocean’s most dynamic job markets."
)
log(f"Site Description: {site_desc}")

# ---------- MISTRAL 7B via GitHub Models (4-bit) ----------
log("Loading mistralai/Mistral-7B-Instruct-v0.2 (4-bit quantized) via GitHub Models...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    trust_remote_code=True
)
model.eval()
log("Mistral-7B-Instruct loaded in 4-bit on CPU")

# ---------- SENTENCE TRANSFORMER ----------
log("Loading all-MiniLM-L6-v2 for semantic deduplication...")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
log("SentenceTransformer ready")

# ---------- PROMPT HELPERS ----------
def mistral_chat_prompt(messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ---------- TITLE GENERATION ----------
def generate_unique_titles(site_desc: str, num_titles: int = 15):
    log(f"Generating {num_titles} SEO-optimized blog titles using Mistral-7B...")
    
    system_msg = "You are a professional SEO blog editor for a Mauritius job portal. Generate diverse, clickable, 7–11 word titles."
    user_msg = (
        f"Generate {num_titles} unique, engaging blog post titles for:\n\"{site_desc}\"\n\n"
        f"Rules:\n"
        f"- 7–11 words each\n"
        f"- Start with action verb, question, or number\n"
        f"- Include Mauritius locations: Ebene, Grand Baie, Port Louis\n"
        f"- Cover: CV tips, interviews, remote work, expat jobs, industry trends\n"
        f"- Return ONLY a numbered list. No explanations."
    )
    
    prompt = mistral_chat_prompt([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ])

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.9,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.25,
            pad_token_id=tokenizer.eos_token_id
        )
    
    raw = tokenizer.decode(output[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
    log(f"Raw Mistral Output:\n{raw}\n")

    # Parse titles
    titles = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line or not line[0].isdigit(): 
            continue
        clean = line.split(". ", 1)[-1].split(": ", 1)[-1].strip(' "\'-')
        words = clean.split()
        if 7 <= len(words) <= 11 and clean[0].isupper():
            titles.append(clean)

    log(f"Extracted {len(titles)} raw candidates")

    # Deduplicate semantically
    unique_titles = []
    embeddings = []
    for title in titles:
        if len(unique_titles) >= num_titles:
            break
        emb = similarity_model.encode(title, convert_to_tensor=True)
        if not embeddings:
            unique_titles.append(title)
            embeddings.append(emb)
            log(f"Title {len(unique_titles)}: {title}")
            continue
        sims = [util.cos_sim(emb, e).item() for e in embeddings]
        if not any(s > 0.88 for s in sims):
            unique_titles.append(title)
            embeddings.append(emb)
            log(f"Title {len(unique_titles)}: {title}")

    # Fill with AI variations
    while len(unique_titles) < num_titles and unique_titles:
        base = random.choice(unique_titles)
        var_prompt = mistral_chat_prompt([
            {"role": "system", "content": "Rephrase completely but keep meaning and SEO value."},
            {"role": "user", "content": f"Rewrite this title differently:\n\"{base}\"\n7–11 words. Start with verb/question. Mauritius job focus."}
        ])
        inputs = tokenizer(var_prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=40, temperature=1.0, do_sample=True, top_p=0.9)
        new_title = tokenizer.decode(out[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
        words = new_title.split()
        if 7 <= len(words) <= 11 and new_title[0].isupper():
            emb = similarity_model.encode(new_title, convert_to_tensor=True)
            sims = [util.cos_sim(emb, e).item() for e in embeddings]
            if not any(s > 0.88 for s in sims):
                unique_titles.append(new_title)
                embeddings.append(emb)
                log(f"Title {len(unique_titles)} (AI variation): {new_title}")

    final = unique_titles[:num_titles]
    log(f"\nFINAL {len(final)} TITLES:\n" + "\n".join([f"{i+1}. {t}" for i, t in enumerate(final)]))
    return final

# ---------- ARTICLE GENERATION ----------
def generate_article(title: str) -> str:
    log(f"\nGenerating full article for:\n→ {title}")
    
    prompt = mistral_chat_prompt([
        {"role": "system", "content": "You are a career expert writing for Mauritius.mimusjobs.com. Write engaging, practical, 550–700 word blog posts with local examples."},
        {"role": "user", "content": (
            f"Write a blog post titled:\n\"{title}\"\n\n"
            f"Include:\n"
            f"- Hook with Mauritius job stat or story\n"
            f"- 4 bullet-point tips with local examples (Ebene, Grand Baie, etc.)\n"
            f"- 1 real success story (e.g. 'Aisha from Quatre Bornes...')\n"
            f"- CTA: 'Find your dream job at Mauritius.mimusjobs.com'\n\n"
            f"550–700 words. Natural tone. Use headers, bullets."
        )}
    ])

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    
    for attempt in range(3):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.8,
                do_sample=True,
                top_p=0.93,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        article = tokenizer.decode(output[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
        word_count = len(article.split())
        
        if word_count >= 550:
            log(f"Article ready: {word_count} words")
            return article
        else:
            log(f"Attempt {attempt+1}: Too short ({word_count} words). Retrying...")

    log("Using last attempt despite length")
    return article

# ---------- MAIN LOOP ----------
try:
    log("="*70)
    log("PHASE 1: GENERATING 15 UNIQUE TITLES")
    log("="*70)
    titles = generate_unique_titles(site_desc, num_titles=15)

    articles = []
    total = len(titles)
    progress = {"total": total, "done": 0, "current": "", "percent": 0}

    log("="*70)
    log("PHASE 2: GENERATING FULL ARTICLES")
    log("="*70)

    for i, title in enumerate(titles, 1):
        progress["current"] = title
        progress["done"] = i - 1
        progress["percent"] = int((i - 1) / total * 100)
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

        content = generate_article(title)
        articles.append({"title": title, "content": content})

        progress["done"] = i
        progress["percent"] = int(i / total * 100)
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

        log(f"PROGRESS: {i}/{total} [{progress['percent']}%] → {title[:50]}...")

    # Save
    with open(articles_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    log(f"SAVED: {articles_file} ({len(articles)} articles)")

    progress["percent"] = 100
    progress["current"] = "Complete!"
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    log("progress.json → 100%")

    print("\n" + "="*70)
    print("SUCCESS: 15 High-Quality Articles Generated with Mistral-7B!")
    print(f"→ Output: {articles_file}")
    print(f"→ Logs: {log_file}")
    print(f"→ Model: mistralai/Mistral-7B-Instruct-v0.2 (GitHub Models)")
    print("="*70)

except Exception as e:
    log(f"ERROR: {str(e)}")
    import traceback
    log(traceback.format_exc())
    print("FAILED")
finally:
    log_handle.close()
