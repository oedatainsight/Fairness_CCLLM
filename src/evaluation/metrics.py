# src/evaluation/metrics.py
import re

def demographic_parity(outputs):
    # example: count 'he' vs 'she'
    male_count = sum("he " in o.lower() for o in outputs)
    female_count = sum("she " in o.lower() for o in outputs)
    return male_count, female_count

def counterfactual_consistency(original_outputs, swapped_outputs):
    """
    Compare original vs. gender-swapped outputs for large divergences in meaning.
    You could measure sentiment difference, etc.
    """
    # naive example
    diffs = []
    for orig, swap in zip(original_outputs, swapped_outputs):
        diffs.append(abs(len(orig) - len(swap)))
    return sum(diffs)/len(diffs)