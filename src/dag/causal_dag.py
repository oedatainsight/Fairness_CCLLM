# src/dag/causal_dag.py
import networkx as nx
import matplotlib.pyplot as plt

def create_causal_dag():
    """
    Define a DAG capturing key variables: Gender, Profession, Bias, Pronoun, Output
    """
    dag = nx.DiGraph()
    dag.add_edges_from([
        ("Gender", "Pronoun"),
        ("Profession", "Pronoun"),
        ("Profession", "Bias"),
        ("Gender", "Bias"),
        ("Bias", "Output")
    ])
    return dag

def visualize_dag():
    dag = create_causal_dag()
    nx.draw(dag, with_labels=True, node_color="lightblue", node_size=3000, font_size=10)
    plt.title("Causal DAG (Gender, Profession, Bias, Output)")
    plt.show()

if __name__ == "__main__":
    visualize_dag()