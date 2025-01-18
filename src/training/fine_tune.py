# src/training/fine_tune.py

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from tqdm import tqdm

from src.embeddings.reweighting import EmbeddingReweighter, apply_reweighting_loss
from src.embeddings.adversarial_training import AdversarialTrainer, Adversary
from src.training.dataset import FairDataset

def fine_tune_model(
    data_csv="data/cf_augmented.csv",
    distribution_csv="data/profession_distribution.csv",
    model_name="gpt2",
    batch_size=2,
    epochs=3
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = FairDataset(data_csv, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Main model
    main_model = AutoModelForCausalLM.from_pretrained(model_name)
    main_model.train()
    
    # Adversary
    adversary = Adversary(embedding_dim=768)  # adjust dim as needed
    adv_trainer = AdversarialTrainer(main_model, adversary, alpha=1.0)

    # Reweighter
    reweighter = EmbeddingReweighter(distribution_csv)

    optimizer = AdamW(main_model.parameters(), lr=5e-5)

    print("Starting fine-tuning...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0  # Track epoch loss

        # Use tqdm for progress tracking
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training Progress")):
            optimizer.zero_grad()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            role = batch["role"]
            cf_gender = batch["cf_gender"]

            labels = input_ids.clone()

            # Adversarial forward pass
            total_loss, main_loss, adv_loss = adv_trainer.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                gender_labels=(cf_gender == "female").long()  # example: map gender to numeric labels
            )
            
            # Reweight main_loss based on role + gender
            w_loss = apply_reweighting_loss(main_loss, role[0], cf_gender[0], reweighter)
            final_loss = w_loss + adv_loss
            final_loss.backward()

            optimizer.step()

            epoch_loss += final_loss.item()

            if batch_idx % 10 == 0:  # Log every 10 batches
                print(f"Batch {batch_idx} Loss: {final_loss.item()}")

        # Log epoch loss
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader)}")

    # Save the final model
    main_model.save_pretrained("results/debiased_model")
    print("Model fine-tuned and saved!")

if __name__ == "__main__":
    fine_tune_model()