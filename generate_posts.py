#!/usr/bin/env python3
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import random
from datetime import datetime

# ---------- CONFIG ----------
log_file = "logs.txt"
progress_file = "progress.json"
articles_file = "articles.json"
log_handle = open(log_file, "a", encoding="utf-8")

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    full = f"[{ts}] {msg}"
    print(full)
    log_handle.write(full + "\n")
    log_handle.flush()

log("AI Blog Generator – Using FLAN-T5-SMALL (77M) – GitHub Optimized")

# ---------- SITE DESCRIPTION ----------
site_desc = (
    "Mauritius.mimusjobs.com is a premier job portal dedicated to connecting talent with opportunities across Mauritius's thriving economy. "
    "From IT roles in Ebene Cybercity to luxury hospitality positions in Grand Baie, the platform features thousands of verified listings in tourism, finance, tech, healthcare, and more. "
    "Job seekers can upload resumes, build ATS-friendly profiles, and receive tailored job alerts, while employers benefit from advanced recruitment tools and company branding. "
    "With a mobile-optimized interface, multilingual support (English, French, Kreol), and AI-powered matching, it empowers locals and expatriates alike to advance their careers in one of the Indian Ocean’s most dynamic job markets."
)
log(f"Site: {site_desc[:100]}...")

# ---------- LOAD TINY MODEL (flan-t5-small) ----------
log("Loading google/flan-t5-small (77M params)...")
device = "cpu"
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()
log("flan-t5-small loaded on CPU")

# ---------- LOAD TINY EMBEDDING MODEL ----------
log("Loading all-MiniLM-L6-v2 (22M)...")
sim_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
log("Embedding model ready")

# ---------- TITLE GENERATION ----------
def generate_titles(num=15):
    log(f"Generating {num} blog titles...")
    prompt = (
        f"List {num} short, catchy blog titles for:\n\"{site_desc}\"\n"
        "Each: 6–10 words. Start with verb or number. "
        "Cover: CV, interview, remote work, Mauritius jobs. "
        "Return only numbered list."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.9,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2
        )
    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    log(f"Raw titles:\n{raw}")

    titles = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line or not line[0].isdigit(): continue
        t = line.split(". ", 1)[-1].strip(' "')
        words = t.split()
        if 6 <= len(words) <= 10 and t[0].isupper():
            titles.append(t)

    # Dedupe
    unique = []
    embs = []
    for t in titles:
        if len(unique) >= num: break
        emb = sim_model.encode(t, convert_to_tensor=True)
        if not embs or all(util.cos_sim(emb, e).item() < 0.85 for e in embs):
            unique.append(t)
            embs.append(emb)
            log(f"Title {len(unique)}: {t}")

    # Fill missing with variations
    while len(unique) < num and unique:
        base = random.choice(unique)
        var = tokenizer.decode(
            model.generate(
                **tokenizer(f"Rephrase: \"{base}\"", return_tensors="pt").to(device),
                max_new_tokens=30,
                temperature=1.0,
                do_sample=True
            )[0],
            skip_special_tokens=True
        ).strip()
        if 6 <= len(var.split()) <= 10 and var[0].isupper():
            emb = sim_model.encode(var, convert_to_tensor=True)
            if all(util.cos_sim(emb, e).item() < 0.85 for e in embs):
                unique.append(var)
                embs.append(emb)
                log(f"Title {len(unique)} (var): {var}")

    final = unique[:num]
    log(f"\nFINAL {len(final)} TITLES:\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(final)))
    return final

# ---------- ARTICLE GENERATION ----------
def generate_article(title):
    log(f"\nWriting article: {title}")
    prompt = (
        f"Write a short blog post titled \"{title}\" for {site_desc}. "
        "Include: intro, 3 tips with examples, 1 success story, conclusion with CTA. "
        "400–600 words. Friendly tone. Use bullet points."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    for _ in range(3):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=700,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.15,
                min_length=300
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        wc = len(text.split())
        if wc >= 400:
            log(f"Done: {wc} words")
            return text
        log(f"Short ({wc} words), retrying...")
    log(f"Using last version ({wc} words)")
    return text

# ---------- MAIN ----------
try:
    log("="*60)
    log("PHASE 1: TITLES")
    log("="*60)
    titles = generate_titles(15)

    articles = []
    total = len(titles)
    prog = {"total": total, "done": 0, "current": "", "percent": 0}

    log("="*60)
    log("PHASE 2: ARTICLES")
    log("="*60)

    for i, t in enumerate(titles, 1):
        prog["current"] = t
        prog["done"] = i - 1
        prog["percent"] = int((i-1)/total*100)
        json.dump(prog, open(progress_file, "w"), indent=2)

        content = generate_article(t)
        articles.append({"title": t, "content": content})

        prog["done"] = i
        prog["percent"] = int(i/total*100)
        json.dump(prog, open(progress_file, "w"), indent=2)
        log(f"{i}/{total} [{prog['percent']}%] → {t[:50]}...")

    json.dump(articles, open(articles_file, "w"), indent=2, ensure_ascii=False)
    prog["percent"] = 100
    prog["current"] = "Done!"
    json.dump(prog, open(progress_file, "w"), indent=2)
    log(f"SAVED: {articles_file} ({len(articles)} articles)")

    print("\n" + "="*60)
    print("SUCCESS! 15 articles ready.")
    print(f"→ {articles_file}")
    print(f"→ {log_file}")
    print("="*60)

except Exception as e:
    log(f"ERROR: {e}")
    import traceback
    log(traceback.format_exc())
    print("FAILED")
finally:
    log_handle.close()
