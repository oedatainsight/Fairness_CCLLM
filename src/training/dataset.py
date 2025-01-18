# src/training/dataset.py
import pandas as pd
from torch.utils.data import Dataset

class FairDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = row["counterfactual_prompt"]
        role = row["role"]
        cf_gender = row["cf_gender"]
        # 'gender_label' can be 0 for male, 1 for female, 2 neutral, etc. if you have numeric labels

        encoded = self.tokenizer(prompt, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()

        # If you're training GPT-2 in a causal manner, you might set labels=input_ids
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "role": role,
            "cf_gender": cf_gender
            # you can store a numeric 'gender_label' if needed
        }