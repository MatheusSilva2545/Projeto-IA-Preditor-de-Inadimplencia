# src/pln.py
"""
Módulo PLN:
- extract_entities(text): usa spaCy pt_core_news_sm para NER
- train_text_classifier(texts, labels): TF-IDF + LogisticRegression (salva vetor e modelo)
- predict_text_class(text): retorna label, probabilidade e entidades
"""
import os
import joblib
from typing import List, Dict

# sklearn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# spaCy
try:
    import spacy
    nlp = spacy.load("pt_core_news_sm")
except Exception:
    nlp = None
    # usuário será instruído a instalar o modelo

VECT_PATH = "models/pln_vectorizer.pkl"
CLF_PATH = "models/pln_clf.pkl"

def extract_entities(text: str) -> List[Dict]:
    if nlp is None:
        return []
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

def train_text_classifier(texts: List[str], labels: List[str]):
    """
    Treina TF-IDF + LogisticRegression.
    texts: lista de strings
    labels: lista de labels (strings)
    Salva modelos em models/
    """
    vec = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, labels)
    os.makedirs("models", exist_ok=True)
    joblib.dump(vec, VECT_PATH)
    joblib.dump(clf, CLF_PATH)
    print("Modelo PLN treinado e salvo.")
    return vec, clf

def predict_text_class(text: str):
    """
    Retorna {label, proba, entities}
    """
    if not os.path.exists(VECT_PATH) or not os.path.exists(CLF_PATH):
        return {"error": "Modelo PLN não treinado. Rode train_text_classifier() primeiro."}
    vec = joblib.load(VECT_PATH)
    clf = joblib.load(CLF_PATH)
    x = vec.transform([text])
    label = clf.predict(x)[0]
    proba = float(clf.predict_proba(x).max())
    ents = extract_entities(text)
    return {"label": label, "proba": proba, "entities": ents}
