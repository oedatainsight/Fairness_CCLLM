# src/evaluation/evaluator.py
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from src.evaluation.metrics import demographic_parity, counterfactual_consistency

 # src/evaluation/evaluator.py
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.metrics import demographic_parity, counterfactual_consistency

def evaluate_fairness(model_path="results/debiased_model", test_csv="data/test_prompts.csv"):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    df = pd.read_csv(test_csv)
    outputs = []
    swapped_outputs = []

    for _, row in df.iterrows():
        prompt = row["prompt"]
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(input_ids, max_length=50)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        outputs.append(text)

        # Example: do a naive "he->she" swap in prompt
        swap_prompt = prompt.replace("he", "she").replace("his", "her")
        swap_ids = tokenizer.encode(swap_prompt, return_tensors="pt")
        with torch.no_grad():
            swap_out = model.generate(swap_ids, max_length=50)
        swapped_text = tokenizer.decode(swap_out[0], skip_special_tokens=True)
        swapped_outputs.append(swapped_text)

    male_count, female_count = demographic_parity(outputs)
    consistency_score = counterfactual_consistency(outputs, swapped_outputs)

    results = {
        "male_pronoun_count": male_count,
        "female_pronoun_count": female_count,
        "counterfactual_consistency": consistency_score
    }
    pd.DataFrame([results]).to_csv("results/fairness_evaluation.csv", index=False)
    print("Evaluation complete, results:", results)

if __name__ == "__main__":
    evaluate_fairness()