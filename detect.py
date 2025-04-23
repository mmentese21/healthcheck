import json
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

nlp = spacy.load("en_core_web_sm")


model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # example binary

def preprocess_chat(chat_history):
    full_text = " ".join(chat_history)
    doc = nlp(full_text)
    clean_text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    
    return clean_text

def predict_disease(chat_text):
    inputs = tokenizer(chat_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)

    pred_class = torch.argmax(probs).item()
    confidence = probs[0][pred_class].item()

    return {
        "disease_detected": bool(pred_class),
        "confidence": round(confidence, 3)
    }

if __name__ == "__main__":
    with open("sample_chats.json", "r") as f:
        chats = json.load(f)

    for user in chats:
        chat_text = preprocess_chat(user["chat_history"])
        result = predict_disease(chat_text)

        print(f"User: {user['user_id']}")
        print(f"→ Disease Detected: {result['disease_detected']} (Confidence: {result['confidence']})")
        print(result)
        print("-" * 40)
