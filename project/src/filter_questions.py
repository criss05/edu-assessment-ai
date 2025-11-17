import pandas as pd
from sentence_transformers import SentenceTransformer, util

# --- Paths ---
QUESTIONS_FOLDER = "./questions"
FILTERED_FOLDER = "./filtered_questions"
INPUT_CSV = f"{QUESTIONS_FOLDER}/lecture12_clean_questions.csv"
OUTPUT_CSV = f"{FILTERED_FOLDER}/lecture12_clean_questions_filtered.csv"

import os
os.makedirs(FILTERED_FOLDER, exist_ok=True)

# --- Load CSV ---
df = pd.read_csv(INPUT_CSV)

# --- Load embedding model ---
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Filtration functions ---

def filter_by_concepts(question, concepts):
    question_lower = str(question).lower()
    for c in concepts:
        if c.lower() in question_lower:
            return True
    return False

def filter_by_length(question, min_len=3, max_len=25):
    words = str(question).split()
    return min_len <= len(words) <= max_len

def is_relevant(sentence, question, threshold=0.6):
    embeddings = embed_model.encode([str(sentence), str(question)], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity >= threshold

# --- Filter loop ---
filtered_questions = []

for _, row in df.iterrows():
    sentence = row['context']
    question = row['question']

    # Extract concepts from sentence (you could also load from KG if available)
    # Here we just split sentence into words as simple approximation
    concepts = sentence.split()  

    if not filter_by_concepts(question, concepts):
        continue
    if not filter_by_length(question):
        continue
    if not is_relevant(sentence, question):
        continue

    filtered_questions.append({
        "context": sentence,
        "question": question
    })

# --- Save filtered CSV ---
filtered_df = pd.DataFrame(filtered_questions)
filtered_df.to_csv(OUTPUT_CSV, index=False)
print(f"[INFO] Saved {len(filtered_df)} filtered questions to {OUTPUT_CSV}")
