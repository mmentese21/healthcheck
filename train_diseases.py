import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.preprocessing import LabelEncoder
import os
import json

df = pd.read_csv("Symptom2Disease.csv")
df["index"] = df["index"].astype(int)


label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])


label2id = {label: int(idx) for idx, label in enumerate(label_encoder.classes_)}
id2label = {v: k for k, v in label2id.items()}

os.makedirs("disease_classifier", exist_ok=True)
with open("disease_classifier/label_map.json", "w") as f:
    json.dump(label2id, f)

train_df = df[df["index"] % 7 != 0].reset_index(drop=True)

print(f"Training samples: {len(train_df)}")

model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)

class DiseaseDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = DiseaseDataset(train_df["text"].tolist(), train_df["label_id"].tolist())

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    do_eval=True,
    save_steps=500
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

model.save_pretrained("disease_classifier")
tokenizer.save_pretrained("disease_classifier")
