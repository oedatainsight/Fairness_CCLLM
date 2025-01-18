# main.py

from src.dag.causal_dag import visualize_dag
from src.dag.counterfactuals import create_augmented_data
from src.training.fine_tune import fine_tune_model
from src.evaluation.evaluator import evaluate_fairness

def main():
    # 1. Construct the Causal Framework
    visualize_dag()  # just to confirm DAG structure

    # 2. Generate or Identify Counterfactuals
    create_augmented_data("data/synthetic_data.csv", "data/cf_augmented.csv")

    # 3. Train/Fine-Tune a Fair LLM
    fine_tune_model(
        data_csv="data/cf_augmented.csv",
        distribution_csv="data/profession_distribution.csv"
    )

    # 4. Evaluate Fairness
    evaluate_fairness(
        model_path="results/debiased_model",
        test_csv="data/test_prompts.csv"
    )

    print("Workflow complete!")

if __name__ == "__main__":
    main()