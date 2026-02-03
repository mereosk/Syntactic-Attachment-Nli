"""Core analysis functions for NLI model evaluation."""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm


def analyze_model_by_phenomenon(model_name, data):
    """Analyze a single NLI model, separating by phenomenon type."""
    print(f"\nAnalyzing {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    label_map = model.config.id2label
    print(f"Label mapping: {label_map}")

    # Find entailment index
    entailment_idx = None
    for idx, label in label_map.items():
        if 'entailment' in label.lower():
            entailment_idx = idx
            break
    if entailment_idx is None:
        entailment_idx = 1

    # Separate data by phenomenon
    temporal_data = [item for item in data if item['type'] == 'temporal']
    instrument_data = [item for item in data if item['type'] == 'Instrument vs Possession']
    serial_data = [item for item in data if item['type'] == 'Serial Attachment']

    results = {}

    for phenom_name, phenom_data in [
        ('Temporal', temporal_data),
        ('Instrument', instrument_data),
        ('Serial', serial_data)
    ]:
        if not phenom_data:
            continue

        print(f"  Processing {phenom_name}: {len(phenom_data)} examples")
        results[phenom_name] = analyze_phenomenon(
            model, tokenizer, phenom_data, phenom_name, entailment_idx
        )

    return results


def analyze_phenomenon(model, tokenizer, data, phenomenon_type, entailment_idx):
    """Analyze a specific phenomenon."""

    js_distances = []
    preference_A_count = 0  # High/Instrument/Nested
    preference_B_count = 0  # Low/Possession/Flat

    control_A_correct = 0  # premise_a with hypothesis_aa
    control_B_correct = 0  # premise_b with hypothesis_bb
    total = 0

    for item in tqdm(data, desc=f"  {phenomenon_type}", leave=False):
        # Baseline ambiguous pairs
        premise = item['premise']

        # Control (unambiguous) pairs
        premise_a = item['premise_a']
        premise_b = item['premise_b']

        pairs = [
            (premise, item['hypothesis_a']),      # 0: baseline A
            (premise, item['hypothesis_b']),      # 1: baseline B
            (premise_a, item['hypothesis_aa']),   # 2: control A correct
            (premise_a, item['hypothesis_ab']),   # 3: control A wrong
            (premise_b, item['hypothesis_ba']),   # 4: control B wrong
            (premise_b, item['hypothesis_bb'])    # 5: control B correct
        ]

        features = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            scores = model(**features).logits
            probs = torch.softmax(scores, dim=1)

        # Get entailment scores
        score_baseline_A = probs[0][entailment_idx].item()
        score_baseline_B = probs[1][entailment_idx].item()
        score_control_A_correct = probs[2][entailment_idx].item()
        score_control_A_wrong = probs[3][entailment_idx].item()
        score_control_B_wrong = probs[4][entailment_idx].item()
        score_control_B_correct = probs[5][entailment_idx].item()

        # Calculate JS divergence for baseline ambiguity
        js_dist = jensenshannon(probs[0].numpy(), probs[1].numpy())
        js_distances.append(js_dist)

        # Count preferences in baseline
        if score_baseline_A > score_baseline_B:
            preference_A_count += 1
        else:
            preference_B_count += 1

        # Check control accuracy
        if score_control_A_correct > score_control_A_wrong:
            control_A_correct += 1

        if score_control_B_correct > score_control_B_wrong:
            control_B_correct += 1

        total += 1

    # Calculate metrics
    avg_js_distance = np.mean(js_distances)
    preference_A_rate = preference_A_count / total
    preference_B_rate = preference_B_count / total
    control_A_acc = control_A_correct / total
    control_B_acc = control_B_correct / total
    accuracy_gap = control_A_acc - control_B_acc
    overall_control_acc = (control_A_correct + control_B_correct) / (2 * total)

    results = {
        'avg_js_distance': avg_js_distance,
        'js_std': np.std(js_distances),
        'js_min': np.min(js_distances),
        'js_max': np.max(js_distances),
        'preference_A_rate': preference_A_rate,
        'preference_B_rate': preference_B_rate,
        'preference_strength': abs(preference_A_rate - 0.5),  # Deviation from balanced
        'control_A_accuracy': control_A_acc,
        'control_B_accuracy': control_B_acc,
        'accuracy_gap': accuracy_gap,
        'overall_accuracy': overall_control_acc,
        'total_examples': total
    }

    return results