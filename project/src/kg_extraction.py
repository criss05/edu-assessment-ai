import spacy
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_triples_with_confidence(text, filename):
    print(f"[DEBUG] Extracting triples from {filename}...")
    doc = nlp(text)
    triples = []
    for sent in doc.sents:
        sent_emb = embed_model.encode(str(sent), convert_to_tensor=True)
        for token in sent:
            if token.dep_ in ("ROOT", "relcl") and token.pos_ == "VERB":
                subjects = [w.text for w in token.lefts if w.dep_ in ("nsubj","nsubjpass")]
                objects = [w.text for w in token.rights if w.dep_ in ("dobj","pobj","attr")]
                for s in subjects:
                    for o in objects:
                        triple_text = f"{s} {token.lemma_} {o}"
                        triple_emb = embed_model.encode(triple_text, convert_to_tensor=True)
                        confidence = util.cos_sim(sent_emb, triple_emb).item()
                        triples.append({
                            "subject": s.strip(),
                            "relation": token.lemma_.strip(),
                            "object": o.strip(),
                            "confidence": confidence,
                            "source_file": filename
                        })
    print(f"[DEBUG] Extracted {len(triples)} triples from {filename}")
    return triples
