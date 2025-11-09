#!/usr/bin/env python3
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
import random
from datetime import datetime
import traceback

# ---------- CONFIG ----------
log_file = "logs.txt"
progress_file = "progress.json"
articles_file = "articles.json"
log_handle = open(log_file, "a", encoding="utf-8")

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    try:
        log_handle.write(full_msg + "\n")
        log_handle.flush()
    except Exception:
        pass

log("AI Blog Generator Started – Using Mistral-7B-Instruct via GitHub Models (4-bit)")

# ---------- HARDCODED SITE DESCRIPTION ----------
site_desc = (
    "Mauritius.mimusjobs.com is a premier job portal dedicated to connecting talent with opportunities across Mauritius's thriving economy. "
    "From IT roles in Ebene Cybercity to luxury hospitality positions in Grand Baie, the platform features thousands of verified listings in tourism, finance, tech, healthcare, and more. "
    "Job seekers can upload resumes, build ATS-friendly profiles, and receive tailored job alerts, while employers benefit from advanced recruitment tools and company branding. "
    "With a mobile-optimized interface, multilingual support (English, French, Kreol), and AI-powered matching, it empowers locals and expatriates alike to advance their careers in one of the Indian Ocean’s most dynamic job markets."
)
log(f"Site Description: {site_desc}")

# ---------- DEVICE / QUANTIZATION CONFIG ----------
use_cuda = torch.cuda.is_available()
if use_cuda:
    log("CUDA detected: will attempt GPU-backed loading.")
    bnb_compute_dtype = torch.float16
    model_dtype = torch.float16
    device_map = "auto"
else:
    log("No CUDA detected: using CPU-safe settings.")
    # On CPU, float16 compute is often unsupported; use float32 for compute dtypes
    bnb_compute_dtype = torch.float32
    model_dtype = torch.float32
    # device_map of 'cpu' works; some quantized loads prefer "cpu"
    device_map = "cpu"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=bnb_compute_dtype,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
log(f"Model ID: {model_id}")

# ---------- TOKENIZER & MODEL LOAD ----------
try:
    log("Loading tokenizer (trust_remote_code=True)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log("Tokenizer loaded.")
except Exception as e:
    log(f"Failed to load tokenizer: {e}")
    log(traceback.format_exc())
    raise

try:
    log("Loading model with quantization config...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=model_dtype,
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    model.eval()
    log(f"Mistral model loaded (device_map={device_map}, dtype={model_dtype}).")
except Exception as e:
    # Provide helpful fallback message and re-raise
    log(f"Model load failed: {e}")
    log(traceback.format_exc())
    raise

# ---------- SENTENCE TRANSFORMER ----------
try:
    st_device = "cuda" if use_cuda else "cpu"
    log(f"Loading SentenceTransformer on {st_device} for semantic deduplication...")
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2", device=st_device)
    log("SentenceTransformer ready")
except Exception as e:
    log(f"Failed to load SentenceTransformer: {e}")
    log(traceback.format_exc())
    raise

# ---------- PROMPT HELPERS ----------
def mistral_chat_prompt(messages):
    """
    Use the tokenizer-provided chat template helper if available.
    Fallback: join roles/content manually.
    """
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # fallback formatting: simple concatenation
        parts = []
        for m in messages:
            role = m.get("role", "user")
            parts.append(f"[{role.upper()}]\n{m.get('content','')}\n")
        # Many instruction-following Mistral adapters expect an instruction wrapper; keep it simple
        return "\n".join(parts)

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
    # Move inputs to appropriate device if model on GPU
    if use_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}

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

    raw = tokenizer.decode(output[0], skip_special_tokens=True)
    # Some model wrappers include special markers; attempt to clean
    if "[/INST]" in raw:
        raw = raw.split("[/INST]")[-1].strip()
    log(f"Raw Mistral Output:\n{raw}\n")

    # Parse titles: accept numbered lines "1. Title"
    titles = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # accept "1." or "1)"
        if line[0].isdigit():
            # remove leading numbering token like "1.", "1)", "1 -"
            parts = line.split(".", 1) if "." in line else line.split(")", 1)
            clean = parts[-1].strip()
        else:
            # if not numbered but looks like a title, keep as fallback
            clean = line
        clean = clean.strip(' "\'–-')
        words = clean.split()
        if 7 <= len(words) <= 11 and clean and clean[0].isupper():
            titles.append(clean)

    log(f"Extracted {len(titles)} raw candidates")

    # Deduplicate semantically
    unique_titles = []
    embeddings = []
    for title in titles:
        if len(unique_titles) >= num_titles:
            break
        try:
            emb = similarity_model.encode(title, convert_to_tensor=True)
        except Exception:
            emb = similarity_model.encode(title)
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

    # Fill with AI variations if needed
    variation_attempts = 0
    while len(unique_titles) < num_titles and unique_titles and variation_attempts < 25:
        base = random.choice(unique_titles)
        var_prompt = mistral_chat_prompt([
            {"role": "system", "content": "Rephrase completely but keep meaning and SEO value."},
            {"role": "user", "content": f"Rewrite this title differently:\n\"{base}\"\n7–11 words. Start with verb/question. Mauritius job focus."}
        ])
        inputs = tokenizer(var_prompt, return_tensors="pt", truncation=True, max_length=256)
        if use_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=40, temperature=1.0, do_sample=True, top_p=0.9)
        new_title = tokenizer.decode(out[0], skip_special_tokens=True)
        if "[/INST]" in new_title:
            new_title = new_title.split("[/INST]")[-1].strip()
        new_title = new_title.strip(' "\'–-')
        words = new_title.split()
        variation_attempts += 1
        if 7 <= len(words) <= 11 and new_title and new_title[0].isupper():
            try:
                emb = similarity_model.encode(new_title, convert_to_tensor=True)
            except Exception:
                emb = similarity_model.encode(new_title)
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
    if use_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}

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
        article = tokenizer.decode(output[0], skip_special_tokens=True)
        if "[/INST]" in article:
            article = article.split("[/INST]")[-1].strip()
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
    print(f"→ Model: {model_id}")
    print("="*70)

except Exception as e:
    log(f"ERROR: {str(e)}")
    log(traceback.format_exc())
    print("FAILED")
finally:
    try:
        log_handle.close()
    except Exception:
        pass
