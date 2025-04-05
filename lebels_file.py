import json
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from spellchecker import SpellChecker
import re

# === Grammar Model ===
grammar_model_name = "textattack/roberta-base-CoLA"
grammar_tokenizer = AutoTokenizer.from_pretrained(grammar_model_name)
grammar_model = AutoModelForSequenceClassification.from_pretrained(grammar_model_name)
grammar_model.eval()

# === Semantic Model ===
semantic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
semantic_labels = ["Meaningful", "Nonsense", "Unrelated"]

# === Spell Checker ===
spell = SpellChecker()

# === NLTK setup ===
nltk.download('punkt')

# === Load JSON and extract text ===
def load_text_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_text = " ".join([page["normal_text"] for page in data if "normal_text" in page and page["normal_text"]])
    return all_text

# === Chunk into sentence groups ===
def chunk_by_sentences(text, max_sentences=4):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    for sentence in sentences:
        current_chunk.append(sentence.strip())
        if len(current_chunk) >= max_sentences:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# === Semantic Meaning Score ===
def semantic_meaning_score(text):
    result = semantic_classifier(text, candidate_labels=semantic_labels)
    return {label: round(score, 4) for label, score in zip(result["labels"], result["scores"])}

# === Full Word Validity Score ===
def full_word_based_score(text):
    words = re.findall(r"\b\w+\b", text.lower())
    valid_words = [word for word in words if word in spell]
    return len(valid_words) / len(words) if words else 0

# === Process everything ===
def process_json_file(json_path, output_path="combined_scores.txt"):
    text_from_json = load_text_from_json(json_path)
    sentence_chunks = chunk_by_sentences(text_from_json, max_sentences=4)

    with open(output_path, "w", encoding="utf-8") as out:
        for i, chunk in enumerate(sentence_chunks):
            semantic_score = semantic_meaning_score(chunk)
            full_word_score = full_word_based_score(chunk)
            meaning_score = semantic_score.get("Meaningful", 0)

            # === Updated Verdict Logic ===
            if full_word_score >= 0.8 and meaning_score >= 0.8:
                verdict = "✅ Very Meaningful"
            elif full_word_score >= 0.6 and meaning_score >= 0.6:
                verdict = "⚠️ Somewhat Meaningful"
            else:
                verdict = "❌ Possibly Not Meaningful"

            out.write(f"Chunk {i+1}:\n{'='*40}\n{chunk}\n")
            out.write(f"Meaningful Score:      {semantic_score['Meaningful']:.4f}\n")
            out.write(f"Nonsense Score:        {semantic_score['Nonsense']:.4f}\n")
            out.write(f"Unrelated Score:       {semantic_score['Unrelated']:.4f}\n")
            out.write(f"Full Word Validity:    {full_word_score:.4f}\n")
            out.write(f"Verdict:               {verdict}\n\n")
            print(f"✅ Processed Chunk {i+1}: {verdict}")

    print(f"\n📝 All scores saved to '{output_path}'")

# === Run the full pipeline ===
if __name__ == "__main__":
    process_json_file("output.json", "combined_scores.txt")
