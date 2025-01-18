# src/dag/counterfactuals.py

import re

def generate_counterfactual(prompt, gender):
    """
    Example: Switch 'he' <-> 'she' to create a gender counterfactual.
    For neutral: replace he/she-> they.
    """
    if gender.lower() == "male":
        # Convert 'she'/'her' to 'he'/'him'
        cf = (prompt.replace("she", "he")
                     .replace("her", "him")
                     .replace("hers", "his"))
    elif gender.lower() == "female":
        # Convert 'he'/'him' to 'she'/'her'
        cf = (prompt.replace("he", "she")
                     .replace("him", "her")
                     .replace("his", "her"))
    else:  # Neutral
        cf = (prompt.replace("he", "they")
                     .replace("she", "they")
                     .replace("him", "them")
                     .replace("her", "them")
                     .replace("his", "their"))
    return cf

def create_augmented_data(input_csv, output_csv):
    """
    Reads input CSV, generates counterfactual examples, and saves an augmented dataset.
    Example input_csv columns: prompt, role, gender
    """
    import pandas as pd
    df = pd.read_csv(input_csv)
    augmented_rows = []

    for _, row in df.iterrows():
        original_prompt = row["prompt"]
        role = row["role"]
        gender = row["gender"]

        # generate male/female/neutral counterfactuals
        for g in ["male", "female", "neutral"]:
            cf_prompt = generate_counterfactual(original_prompt, g)
            augmented_rows.append({
                "original_prompt": original_prompt,
                "counterfactual_prompt": cf_prompt,
                "role": role,
                "original_gender": gender,
                "cf_gender": g
            })

    pd.DataFrame(augmented_rows).to_csv(output_csv, index=False)
    print(f"Augmented data saved to {output_csv}")

if __name__ == "__main__":
    create_augmented_data("data/synthetic_data.csv", "data/cf_augmented.csv")