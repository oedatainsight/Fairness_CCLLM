# src/embeddings/reweighting.py
import pandas as pd

class EmbeddingReweighter:
    """
    Computes weights based on how underrepresented a (role, gender) combination is.
    """
    def __init__(self, distribution_csv):
        """
        distribution_csv columns might be:
        Role, Gender, Frequency
        e.g. 'CEO', 'female', 0.2
        """
        dist_df = pd.read_csv(distribution_csv)
        self.weights = {}
        for _, row in dist_df.iterrows():
            role, gender, freq = row["Role"], row["Gender"], float(row["Frequency"])
            self.weights[(role.lower(), gender.lower())] = 1.0 / (freq + 1e-6)

    def get_weight(self, role, gender):
        return self.weights.get((role.lower(), gender.lower()), 1.0)

def apply_reweighting_loss(loss, role, gender, reweighter):
    w = reweighter.get_weight(role, gender)
    return loss * w