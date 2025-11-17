# hybrid_pipeline.py
import os
import pandas as pd
from tqdm import tqdm
import spacy
from transformers import pipeline
import nltk
from nltk import sent_tokenize

nltk.download("punkt")

# Paths
PROCESSED_FOLDER = "./processed"
KG_OUTPUT_FILE = "./knowledge_graph/kg_triples.csv"
QUESTIONS_FOLDER = "./questions"

os.makedirs(QUESTIONS_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(KG_OUTPUT_FILE), exist_ok=True)

# Load spaCy model for KG extraction
nlp = spacy.load("en_core_web_sm")

# Load FLAN-T5 large for question generation (much better quality)
print("[INFO] Loading question-generation model: FLAN-T5-large...")
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=-1  # CPU
)

# --- Step 1: Knowledge Graph Extraction (from all files) ---
triples = []

print("[INFO] Extracting knowledge graph from all files...")
txt_files = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith(".txt")]

for i, filename in enumerate(txt_files):
    print(f"[INFO] Generating questions for file {i+1}/{len(txt_files)}: {filename}")

    filepath = os.path.join(PROCESSED_FOLDER, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    doc = nlp(text)

    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ("ROOT", "relcl") and token.pos_ == "VERB":
                subject = [w.text for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
                objects = [w.text for w in token.rights if w.dep_ in ("dobj", "pobj", "attr")]

                if subject and objects:
                    for s in subject:
                        for o in objects:
                            triples.append({
                                "subject": s.strip(),
                                "relation": token.lemma_.strip(),
                                "object": o.strip(),
                                "source_file": filename
                            })

# Save combined KG
kg_df = pd.DataFrame(triples)
kg_df.to_csv(KG_OUTPUT_FILE, index=False)
print(f"[INFO] Saved KG with {len(kg_df)} triples to {KG_OUTPUT_FILE}")

# --- Step 2: Question Generation (per file) ---
print("[INFO] Generating questions for each file...")

for filename in tqdm(os.listdir(PROCESSED_FOLDER)):
    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(PROCESSED_FOLDER, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = sent_tokenize(text)
    generated = []

    # KG concepts relevant to this file
    file_kg = kg_df[kg_df["source_file"] == filename]
    concepts = set(file_kg['subject'].tolist() + file_kg['object'].tolist())

    print(f"[INFO] Processing {len(sentences)} sentences from {filename}")

    for index, sent in enumerate(sentences, start=1):
        relevant_concepts = [c for c in concepts if c.lower() in sent.lower()]
        print("sent:", sent)

        base_prompt = (
            "Generate one clear factual study question based strictly on the sentence below. "
            "Do NOT create multiple-choice questions. "
            'Do NOT use phrases like "Which of the following". '
            "Do NOT ask about incorrect options. "
            "Ask a simple direct question.\n"
            f"Sentence: \"{sent}\""
        )

        if relevant_concepts:
            prompt = (
                base_prompt +
                f"\nRequired concepts to include in the question: {', '.join(relevant_concepts)}"
            )
        else:
            prompt = base_prompt

        # Generate question using FLAN-T5 (no return_full_text!!)
        outputs = generator(
            prompt,
            max_new_tokens=128,
            num_return_sequences=1
        )

        question_text = outputs[0]['generated_text'].strip()
        print("question_text: ", question_text)

        # Optional: only keep first line
        question_text = question_text.split("\n")[0].strip()
        print("question_text: ", question_text)

        generated.append({
            "context": sent,
            "question": question_text
        })

        print(f"[INFO] Processed {index}/{len(sentences)} sentences...")

    # Save questions CSV
    output_csv = os.path.join(QUESTIONS_FOLDER, filename.replace(".txt", "_questions.csv"))
    pd.DataFrame(generated).to_csv(output_csv, index=False)
    print(f"[INFO] Saved {len(generated)} questions for {filename} to {output_csv}")
