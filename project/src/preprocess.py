"""
preprocess.py
-------------
Module for reading educational materials (PDFs) and preprocessing text
for intelligent knowledge extraction and question generation.

Steps:
1. Extract text from PDF files
2. Clean and normalize text
3. Perform linguistic preprocessing using spaCy:
   - Sentence segmentation
   - Tokenization
   - Lemmatization
   - Stopword removal
   - POS tagging and NER
4. Save the cleaned text for further processing

Dependencies:
- PyMuPDF (fitz)
- spaCy (en_core_web_sm)
"""

import os
import re
import fitz  # PyMuPDF
import spacy


# -----------------------------
# 1. Load spaCy model
# -----------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# -----------------------------
# 2. Extract text from PDFs
# -----------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts raw text from a PDF file using PyMuPDF.
    """
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text")
    except Exception as e:
        print(f"[ERROR] Failed to read {pdf_path}: {e}")
    return text


def extract_all_pdfs(pdf_folder: str) -> dict:
    """
    Extracts text from all PDFs in a given folder.
    Returns a dictionary: {filename: text}
    """
    pdf_texts = {}
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            path = os.path.join(pdf_folder, file)
            pdf_texts[file] = extract_text_from_pdf(path)
            print(f"[INFO] Extracted text from {file}")
    return pdf_texts


# -----------------------------
# 3. Text Cleaning
# -----------------------------
def clean_text(text: str) -> str:
    """
    Removes unwanted characters, line breaks, and extra spaces.
    """
    text = re.sub(r'\s+', ' ', text)        # collapse whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII chars
    text = re.sub(r'\[[0-9]+\]', '', text)  # remove reference numbers like [1]
    text = text.strip()
    return text


# -----------------------------
# 4. Linguistic Preprocessing
# -----------------------------
def preprocess_text(text: str) -> list:
    """
    Tokenizes and lemmatizes text while removing stopwords and punctuation.
    Returns a list of processed sentences.
    """
    doc = nlp(text)
    sentences = []

    for sent in doc.sents:
        tokens = [
            token.lemma_.lower()
            for token in sent
            if not token.is_stop and token.is_alpha
        ]
        if tokens:
            sentences.append(" ".join(tokens))

    return sentences


# -----------------------------
# 5. Full Pipeline
# -----------------------------
def preprocess_pdfs(pdf_folder: str, output_folder: str = "processed"):
    """
    Full preprocessing pipeline for all PDFs in a folder.
    Saves cleaned, tokenized sentences in text files.
    """
    os.makedirs(output_folder, exist_ok=True)

    pdf_texts = extract_all_pdfs(pdf_folder)

    for name, raw_text in pdf_texts.items():
        cleaned = clean_text(raw_text)
        sentences = preprocess_text(cleaned)

        output_path = os.path.join(output_folder, name.replace(".pdf", "_clean.txt"))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sentences))

        print(f"[INFO] Preprocessed {name} → {output_path}")


# -----------------------------
# 6. Run Example
# -----------------------------
if __name__ == "__main__":
    pdf_input_folder = "data"        # e.g. folder containing your 5 PDF files
    output_folder = "processed"      # where preprocessed .txt files will be saved
    preprocess_pdfs(pdf_input_folder, output_folder)
