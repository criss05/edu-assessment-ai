import pandas as pd
from sentence_transformers import SentenceTransformer
from langdetect import detect

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def filter_questions(df):
    seen = set()
    filtered = []
    for _, row in df.iterrows():
        q = row['question'].strip()
        if not q or q in seen:
            continue
        try:
            if detect(q) != 'en':
                continue
        except:
            pass
        if len(q.split()) < 3 or len(q.split()) > 25:
            continue
        seen.add(q)
        filtered.append({"context": row['context'], "question": q, "type": row.get('type','')})
    print(f"[INFO] Filtered questions: {len(filtered)} / {len(df)}")
    return pd.DataFrame(filtered)
