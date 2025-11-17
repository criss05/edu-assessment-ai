import os
import pandas as pd
from kg_extraction import extract_triples_with_confidence
from generation import generate_question, plan_question_type
from filtering import filter_questions

PROCESSED_FOLDER = "./processed"
QUESTIONS_FOLDER = "./questions"
FILTERED_FOLDER = "./filtered_questions"
os.makedirs(QUESTIONS_FOLDER, exist_ok=True)
os.makedirs(FILTERED_FOLDER, exist_ok=True)

# Step 1: Knowledge Graph Extraction
all_triples = []
txt_files = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith(".txt")]
for f in txt_files:
    path = os.path.join(PROCESSED_FOLDER, f)
    text = open(path, "r", encoding="utf-8").read()
    triples = extract_triples_with_confidence(text, f)
    all_triples.extend(triples)

kg_df = pd.DataFrame(all_triples)
kg_df.to_csv("./knowledge_graph/kg_triples.csv", index=False)

# Step 2: Question Generation per file
for f in txt_files:
    path = os.path.join(PROCESSED_FOLDER, f)
    text = open(path, "r", encoding="utf-8").read()
    
    # Split text into 3-sentence paragraphs
    sentences = text.split(". ")
    paragraphs = [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
    
    questions = []
    for para in paragraphs:
        relevant_concepts = [c for c in kg_df['subject'].tolist() + kg_df['object'].tolist() if c.lower() in para.lower()]
        question_text = generate_question(para, relevant_concepts, kg_df)
        questions.append({"context": para, "question": question_text, "type": plan_question_type(para)})
    
    df_q = pd.DataFrame(questions)
    
    # Step 3: Filtering
    df_q_filtered = filter_questions(df_q)
    df_q_filtered.to_csv(os.path.join(FILTERED_FOLDER, f.replace(".txt", "_filtered.csv")), index=False)
