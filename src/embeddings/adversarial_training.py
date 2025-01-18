# src/embeddings/adversarial_training.py
import torch
import torch.nn as nn

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class Adversary(nn.Module):
    """
    A small classifier to predict gender from embeddings.
    """
    def __init__(self, embedding_dim=768, hidden_dim=256, num_labels=2):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class AdversarialTrainer:
    """
    Wraps a main model and an adversary to do adversarial debiasing.
    """
    def __init__(self, main_model, adversary, alpha=1.0):
        self.main_model = main_model
        self.adversary = adversary
        self.alpha = alpha
        self.adv_loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels, gender_labels):
        # Forward pass in main model
        outputs = self.main_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        main_loss = outputs.loss

        # Suppose 'outputs.hidden_states[-1]' are the final layer states; pick [CLS] or first token
        # (For GPT2, you'd adapt accordingly)
        embeddings = outputs.hidden_states[-1][:, 0, :]

        # Reverse gradient
        reversed_emb = GradientReversal.apply(embeddings, self.alpha)
        adv_logits = self.adversary(reversed_emb)
        adv_loss = self.adv_loss_fn(adv_logits, gender_labels)

        total_loss = main_loss + adv_loss
        return total_loss, main_loss, adv_loss