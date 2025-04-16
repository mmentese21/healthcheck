import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

# === Load label map ===
with open("disease_classifier/label_map.json", "r") as f:
    label2id = json.load(f)
id2label = {int(v): k for k, v in label2id.items()}

# === Load tokenizer and model ===
choose = input("Choose model (1 for Bio_ClinicalBERT, 2 for custom): ")
if choose == "2":
    model = AutoModelForSequenceClassification.from_pretrained("disease_classifier")
    tokenizer = AutoTokenizer.from_pretrained("disease_classifier")
elif choose == "1":
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    )
else:
    print("Invalid choice. Exiting.")
    exit()

model.eval()

# === Load and filter test data ===
df = pd.read_csv("Symptom2Disease.csv")  # Replace with your CSV file
df["index"] = df["index"].astype(int)
test_df = df[df["index"] % 7 == 0].reset_index(drop=True)

# === Prediction function ===
def predict_disease(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    return id2label[pred], round(probs[0][pred].item(), 3)

# === Run predictions on test data ===
true_count = 0
total_count = len(test_df)
for i, row in test_df.iterrows():
    user_index = row["index"]
    true_label = row["label"]
    text = row["text"]
    
    predicted_label, confidence = predict_disease(text)
    if predicted_label == true_label:
        true_count += 1
    
    with open("predictions.txt", "a+") as f:
        f.write(f"{user_index},{true_label},{predicted_label},{confidence}\n")
"""
    print(f"Index: {user_index}")
    print(f"Text: {text}")
    print(f"True Label: {true_label}")
    print(f"Predicted: {predicted_label} (Confidence: {confidence})")
    print("-" * 60)
"""

print(f"Total samples: {total_count}")
print(f"Correct predictions: {true_count}")