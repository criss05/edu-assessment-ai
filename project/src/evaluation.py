import pandas as pd
from datasets import load_metric

def evaluate_questions(reference_csv, generated_csv):
    df_ref = pd.read_csv(reference_csv)
    df_gen = pd.read_csv(generated_csv)
    
    references = df_ref['context'].tolist()
    predictions = df_gen['question'].tolist()

    rouge = load_metric("rouge")
    rouge_score = rouge.compute(predictions=predictions, references=references)
    
    bertscore = load_metric("bertscore")
    bert_score = bertscore.compute(predictions=predictions, references=references, lang="en")

    print("[INFO] ROUGE:", rouge_score)
    print("[INFO] BERTScore:", bert_score)
