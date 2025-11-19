import pandas as pd
import glob
import evaluate

# -----------------------------
# Step 1: Combine all generated question CSVs
# -----------------------------
question_files = glob.glob("filtered_questions/*.csv")  # adjust path
df_list = [pd.read_csv(f) for f in question_files]
df_questions = pd.concat(df_list, ignore_index=True)

# -----------------------------
# Step 2: Filter duplicates if needed
# -----------------------------
df_questions_filtered = df_questions.drop_duplicates(subset=['question'])

# -----------------------------
# Step 3: Count question types
# -----------------------------
total_questions = len(df_questions)
questions_retained = len(df_questions_filtered)
open_ended = len(df_questions_filtered[df_questions_filtered['type'] == 'open-ended'])
true_false = len(df_questions_filtered[df_questions_filtered['type'] == 'true/false'])
fill_in_blank = len(df_questions_filtered[df_questions_filtered['type'] == 'fill-in-the-blank'])

# -----------------------------
# Step 4: Compute ROUGE-L and BERTScore-F1
# -----------------------------
references = df_questions_filtered['context'].tolist()
predictions = df_questions_filtered['question'].tolist()

# ROUGE
rouge = evaluate.load("rouge")
rouge_score = rouge.compute(predictions=predictions, references=references)
avg_rouge_l = rouge_score['rougeL']

# BERTScore
bertscore = evaluate.load("bertscore")
bert_score = bertscore.compute(predictions=predictions, references=references, lang="en")
avg_bertscore_f1 = sum(bert_score['f1']) / len(bert_score['f1'])

# -----------------------------
# Step 5: Combine all KG extraction CSVs
# -----------------------------
kg_files = glob.glob("knowledge_graph/*.csv")  # adjust path
df_kg_list = [pd.read_csv(f) for f in kg_files]
df_kg = pd.concat(df_kg_list, ignore_index=True)

triples_extracted = len(df_kg)
avg_confidence = df_kg['confidence'].mean()

# -----------------------------
# Step 6: Pseudo precision/recall/F1 for KG
# -----------------------------
# High-confidence triples considered "correct" (>=0.5)
tp = sum(df_kg['confidence'] >= 0.5)
fp = sum(df_kg['confidence'] < 0.5)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / triples_extracted if triples_extracted > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# -----------------------------
# Step 7: Print summary table
# -----------------------------
summary = {
    "TOTAL QUESTIONS GENERATED": total_questions,
    "QUESTIONS RETAINED AFTER FILTERING": questions_retained,
    "OPEN-ENDED QUESTIONS": open_ended,
    "TRUE/FALSE QUESTIONS": true_false,
    "FILL-IN-THE-BLANK QUESTIONS": fill_in_blank,
    "AVERAGE ROUGE-L": avg_rouge_l,
    "AVERAGE BERTScore-F1": avg_bertscore_f1,
    "TRIPLES EXTRACTED": triples_extracted,
    "AVERAGE CONFIDENCE": avg_confidence,
    "PRECISION": precision,
    "RECALL": recall,
    "F1-SCORE": f1
}

print("\n=== EVALUATION SUMMARY ===")
for k, v in summary.items():
    print(f"{k}: {v}")
