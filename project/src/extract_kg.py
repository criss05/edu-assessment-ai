"""
extract_kg.py
-------------
Lightweight Knowledge Graph extraction prototype.

Input:
 - 'processed/' folder containing preprocessed text files (one sentence per line)

Output:
 - kg_triples.csv       : CSV with columns [subject, relation, object, confidence, source_file, sentence]
 - kg_graph.gml         : Graph saved in GML format
 - kg_graph.png         : Simple visualization image (optional)

Notes:
 - This is a prototype (rule-based + heuristics). For higher accuracy you can
   replace relation extraction with supervised models (RE) or OpenIE systems.
"""

import os
import csv
import re
from collections import defaultdict
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def load_processed_texts(processed_folder="processed"):
    """
    Load processed .txt files. Returns dict: {filename: [sentences...]}
    """
    data = {}
    for fname in sorted(os.listdir(processed_folder)):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(processed_folder, fname)
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            if lines:
                data[fname] = lines
    return data


def normalize_entity(span_text):
    """
    Basic normalization of entity text: lowercase, collapse spaces, strip.
    Could be extended with lemmatization or mapping.
    """
    s = re.sub(r'\s+', ' ', span_text.strip().lower())
    return s


def extract_entities_from_sentence(doc):
    """
    Given a spaCy Doc for a sentence, return a list of entity spans (text, start_idx, end_idx)
    We use named entities and noun chunks as candidate concepts.
    """
    ents = []
    # Named entities
    for ent in doc.ents:
        ents.append((normalize_entity(ent.text), ent.start_char, ent.end_char, "NER"))
    # Noun chunks (fallback / complement)
    for nc in doc.noun_chunks:
        txt = normalize_entity(nc.text)
        # avoid duplicates (NER already captured)
        if not any(txt == e[0] for e in ents):
            ents.append((txt, nc.start_char, nc.end_char, "NOUN_CHUNK"))
    return ents


def extract_svo_triples(doc, ents_in_sent):
    """
    Extract (subject, verb, object) triples using dependency labels.
    Returns list of (subj_text, verb_lemma, obj_text, confidence)
    """
    triples = []
    # Map token indexes to entity texts (if token inside an entity span)
    token_idx_to_ent = {}
    for ent_text, start_char, end_char, typ in ents_in_sent:
        # find tokens that match span
        for token in doc:
            if token.idx >= start_char and token.idx + len(token.text) <= end_char:
                token_idx_to_ent[token.i] = ent_text

    for token in doc:
        # Look for verbs with nominal subject and object (nsubj + dobj)
        if token.pos_ == "VERB":
            subj = None
            dobj = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj = token_idx_to_ent.get(child.i, child.text)
                if child.dep_ in ("dobj", "obj"):
                    dobj = token_idx_to_ent.get(child.i, child.text)
            if subj and dobj:
                s = normalize_entity(subj)
                o = normalize_entity(dobj)
                rel = token.lemma_.lower()
                triples.append((s, rel, o, 1.0))  # high confidence
    return triples


def extract_prep_triples(doc, ents_in_sent):
    """
    Try to extract relations like 'X is a type of Y' or prepositional relations:
    e.g., 'X of Y', 'X in Y' -> map to relation 'in' or 'of' with medium confidence
    """
    triples = []
    token_idx_to_ent = {}
    for ent_text, start_char, end_char, typ in ents_in_sent:
        for token in doc:
            if token.idx >= start_char and token.idx + len(token.text) <= end_char:
                token_idx_to_ent[token.i] = ent_text

    for token in doc:
        if token.dep_ == "prep":
            pobj = None
            pobj_ent = None
            # find object of the preposition
            for child in token.children:
                if child.dep_ == "pobj":
                    pobj_ent = token_idx_to_ent.get(child.i, child.text)
                    pobj = normalize_entity(pobj_ent)
            # the head of prep often connects to a noun (the left-hand entity)
            head = token.head
            head_ent = token_idx_to_ent.get(head.i, head.text)
            if head_ent and pobj:
                s = normalize_entity(head_ent)
                o = pobj
                rel = token.text.lower()  # e.g., 'in', 'of', 'for'
                triples.append((s, rel, o, 0.7))  # medium confidence
    return triples


def extract_cooccurrence_triples(sent, ents_in_sent):
    """
    Fallback: if two entities co-occur in the same sentence and no SVO/prep triple found,
    create a 'related_to' triple with lower confidence.
    """
    triples = []
    ent_texts = [normalize_entity(e[0]) for e in ents_in_sent]
    # create pairwise relations
    for i in range(len(ent_texts)):
        for j in range(i + 1, len(ent_texts)):
            s = ent_texts[i]
            o = ent_texts[j]
            # avoid trivial identical pairs
            if s != o:
                triples.append((s, "related_to", o, 0.4))
    return triples


def unique_triple_key(t):
    """Key for deduplication: subject|relation|object"""
    s, r, o, conf = t
    return f"{s}|{r}|{o}"


def extract_triples_from_sentence(sentence):
    """
    High-level pipeline for a single sentence string:
    - parse with spaCy
    - extract entity spans
    - apply SVO extraction, prep extraction, fallback co-occurrence
    - return list of triples with confidences
    """
    doc = nlp(sentence)
    ents = extract_entities_from_sentence(doc)
    triples = []
    # Strong SVO triples
    triples.extend(extract_svo_triples(doc, ents))
    # Prepositional triples
    triples.extend(extract_prep_triples(doc, ents))
    # If nothing found and we have at least 2 entities, add co-occurrence triples
    if not triples and len(ents) >= 2:
        triples.extend(extract_cooccurrence_triples(sentence, ents))
    # Deduplicate preserving max confidence
    dedup = {}
    for t in triples:
        key = unique_triple_key(t)
        if key not in dedup or t[3] > dedup[key][3]:
            dedup[key] = t
    return list(dedup.values()), ents, doc.text


def extract_kg_from_processed(processed_folder="processed", out_csv="kg_triples.csv",
                              out_gml="kg_graph.gml", out_png="kg_graph.png"):
    """
    Process all files in processed_folder and extract triples.
    Saves CSV and graph files.
    """
    data = load_processed_texts(processed_folder)
    all_triples = []
    for fname, sentences in data.items():
        for sent in sentences:
            triples, ents, sentence_text = extract_triples_from_sentence(sent)
            for (s, r, o, conf) in triples:
                all_triples.append({
                    "subject": s,
                    "relation": r,
                    "object": o,
                    "confidence": conf,
                    "source_file": fname,
                    "sentence": sentence_text
                })

    # Save CSV
    if all_triples:
        df = pd.DataFrame(all_triples)
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[INFO] Saved {len(all_triples)} triples to {out_csv}")
    else:
        print("[WARN] No triples extracted.")

    # Build graph
    G = nx.DiGraph()
    for t in all_triples:
        s = t["subject"]
        o = t["object"]
        r = t["relation"]
        conf = t["confidence"]
        if not G.has_node(s):
            G.add_node(s, label=s)
        if not G.has_node(o):
            G.add_node(o, label=o)
        # Use relation as edge label; store max confidence if multiple edges occur
        if G.has_edge(s, o):
            # update attributes
            prev_conf = G[s][o].get("confidence", 0.0)
            if conf > prev_conf:
                G[s][o]["relation"] = r
                G[s][o]["confidence"] = conf
        else:
            G.add_edge(s, o, relation=r, confidence=conf)

    # Save graph GML
    nx.write_gml(G, out_gml)
    print(f"[INFO] Saved graph to {out_gml}")

    # Try to draw graph (may be small)
    try:
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G, k=0.5)
        labels = {n: n for n in G.nodes()}
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color="lightblue")
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        edge_labels = nx.get_edge_attributes(G, "relation")
        nx.draw_networkx_edges(G, pos, arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[INFO] Graph visualization saved to {out_png}")
    except Exception as e:
        print(f"[WARN] Could not draw graph visualization: {e}")

    return all_triples, G


if __name__ == "__main__":
    triples, G = extract_kg_from_processed()
