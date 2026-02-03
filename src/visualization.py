"""Visualization functions for analysis results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from .config import FIGURES_DIR, TABLES_DIR


def create_comprehensive_visualizations(all_model_results, models_to_test):
    """Create publication-ready visualizations."""

    phenomena = ['Temporal', 'Instrument', 'Serial']

    # === FIGURE 1: Main 3-panel comparison across phenomena ===
    fig1, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    model_names_short = [m.split('/')[-1] for m in models_to_test]

    # Panel A: JS Distance (Uncertainty)
    for i, phenom in enumerate(phenomena):
        js_values = []
        for model in models_to_test:
            if phenom in all_model_results[model]:
                js_values.append(all_model_results[model][phenom]['avg_js_distance'])
            else:
                js_values.append(0)

        x_pos = np.arange(len(models_to_test)) + i * 0.25
        axes[0].bar(x_pos, js_values, width=0.25, label=phenom, color=colors[i], alpha=0.8)

    axes[0].set_xticks(np.arange(len(models_to_test)) + 0.25)
    axes[0].set_xticklabels(model_names_short, rotation=45, ha='right')
    axes[0].set_ylabel('JS Distance', fontsize=12)
    axes[0].set_title('(A) Baseline Uncertainty (JS Distance)', fontsize=13, fontweight='bold')
    axes[0].legend(frameon=True, fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='High Uncertainty')

    # Panel B: Preference Strength (Deviation from 0.5)
    for i, phenom in enumerate(phenomena):
        pref_strengths = []
        for model in models_to_test:
            if phenom in all_model_results[model]:
                pref_strengths.append(all_model_results[model][phenom]['preference_strength'])
            else:
                pref_strengths.append(0)

        x_pos = np.arange(len(models_to_test)) + i * 0.25
        axes[1].bar(x_pos, pref_strengths, width=0.25, label=phenom, color=colors[i], alpha=0.8)

    axes[1].set_xticks(np.arange(len(models_to_test)) + 0.25)
    axes[1].set_xticklabels(model_names_short, rotation=45, ha='right')
    axes[1].set_ylabel('|Preference - 0.5|', fontsize=12)
    axes[1].set_title('(B) Preference Bias Strength', fontsize=13, fontweight='bold')
    axes[1].legend(frameon=True, fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.3, label='Balanced')

    # Panel C: Accuracy Gap
    for i, phenom in enumerate(phenomena):
        gaps = []
        for model in models_to_test:
            if phenom in all_model_results[model]:
                gaps.append(all_model_results[model][phenom]['accuracy_gap'])
            else:
                gaps.append(0)

        x_pos = np.arange(len(models_to_test)) + i * 0.25
        axes[2].bar(x_pos, gaps, width=0.25, label=phenom, color=colors[i], alpha=0.8)

    axes[2].set_xticks(np.arange(len(models_to_test)) + 0.25)
    axes[2].set_xticklabels(model_names_short, rotation=45, ha='right')
    axes[2].set_ylabel('Accuracy Gap (A - B)', fontsize=12)
    axes[2].set_title('(C) Disambiguation Asymmetry', fontsize=13, fontweight='bold')
    axes[2].legend(frameon=True, fontsize=10)
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.3, label='Balanced')

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "main_phenomenon_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    print("Saved: main_phenomenon_comparison.png")

    # === FIGURE 2: Detailed per-phenomenon analysis ===
    for phenom in phenomena:
        fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig2.suptitle(f'{phenom} Attachment Analysis', fontsize=16, fontweight='bold')

        # Collect data for this phenomenon
        js_dists = []
        pref_A_rates = []
        pref_B_rates = []
        acc_A = []
        acc_B = []

        for model in models_to_test:
            if phenom in all_model_results[model]:
                r = all_model_results[model][phenom]
                js_dists.append(r['avg_js_distance'])
                pref_A_rates.append(r['preference_A_rate'])
                pref_B_rates.append(r['preference_B_rate'])
                acc_A.append(r['control_A_accuracy'])
                acc_B.append(r['control_B_accuracy'])

        # Top-left: JS Distance
        axes[0, 0].bar(range(len(model_names_short)), js_dists, color='steelblue')
        axes[0, 0].set_xticks(range(len(model_names_short)))
        axes[0, 0].set_xticklabels(model_names_short, rotation=45, ha='right')
        axes[0, 0].set_ylabel('JS Distance')
        axes[0, 0].set_title('Average JS Distance (Lower = More Confident)')
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Top-right: Attachment Preferences
        x = np.arange(len(model_names_short))
        width = 0.35

        # Label based on phenomenon type
        if phenom == 'Temporal':
            label_A, label_B = 'High (V1)', 'Low (V2)'
        elif phenom == 'Instrument':
            label_A, label_B = 'Instrument', 'Possession'
        else:  # Serial
            label_A, label_B = 'Nested', 'Flat'

        axes[0, 1].bar(x - width/2, pref_A_rates, width, label=label_A, color='coral')
        axes[0, 1].bar(x + width/2, pref_B_rates, width, label=label_B, color='lightblue')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names_short, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Preference Rate')
        axes[0, 1].set_title('Baseline Preferences')
        axes[0, 1].legend()
        axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Balanced')
        axes[0, 1].set_ylim([0, 1])

        # Bottom-left: Overall Control Accuracy
        overall_acc = [(a + b) / 2 for a, b in zip(acc_A, acc_B)]
        axes[1, 0].bar(range(len(model_names_short)), overall_acc, color='mediumseagreen')
        axes[1, 0].set_xticks(range(len(model_names_short)))
        axes[1, 0].set_xticklabels(model_names_short, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Overall Control Accuracy')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Bottom-right: Control Accuracy by Type
        axes[1, 1].bar(x - width/2, acc_A, width, label=f'{label_A} Control', color='orange')
        axes[1, 1].bar(x + width/2, acc_B, width, label=f'{label_B} Control', color='purple')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names_short, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Control Accuracy by Type')
        axes[1, 1].legend()
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"{phenom.lower()}_detailed_analysis.png",
            dpi=300,
            bbox_inches="tight"
        )
        print(f"Saved: {phenom.lower()}_detailed_analysis.png")
        plt.close()

    plt.show()



def create_summary_table(all_model_results, models_to_test):
    """Create a comprehensive summary table."""

    phenomena = ['Temporal', 'Instrument', 'Serial']

    summary_data = []

    for model in models_to_test:
        for phenom in phenomena:
            if phenom in all_model_results[model]:
                r = all_model_results[model][phenom]
                summary_data.append({
                    'Model': model.split('/')[-1],
                    'Phenomenon': phenom,
                    'JS_Distance': r['avg_js_distance'],
                    'Pref_A_Rate': r['preference_A_rate'],
                    'Pref_Strength': r['preference_strength'],
                    'Control_A_Acc': r['control_A_accuracy'],
                    'Control_B_Acc': r['control_B_accuracy'],
                    'Accuracy_Gap': r['accuracy_gap'],
                    'Overall_Acc': r['overall_accuracy']
                })

    df = pd.DataFrame(summary_data)

    print("\n" + "="*100)
    print("COMPREHENSIVE RESULTS BY PHENOMENON")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)

    # Save to CSV
    output_path = TABLES_DIR / "phenomenon_comparison_results.csv"
    df.to_csv(output_path, index=False)
    print("\nResults saved to 'phenomenon_comparison_results.csv'")

    return df


def print_interpretation_guide():
    """Print guide for interpreting results."""
    print("\n" + "="*100)
    print("INTERPRETATION GUIDE")
    print("="*100)
    print("""
JS Distance (Baseline Uncertainty):
  - Lower (0.0-0.3): Model is CONFIDENT about one interpretation (shows bias)
  - Medium (0.3-0.5): Model is somewhat uncertain
  - Higher (0.5+): Model treats baseline as truly ambiguous
  → LOW values indicate models DON'T model ambiguity well

Preference Strength:
  - Deviation from 0.5 (balanced preference)
  - Higher values = stronger systematic bias
  - 0.0 = perfectly balanced, 0.5 = completely one-sided

Accuracy Gap (Control A - Control B):
  - Positive: Better at A-type disambiguation
  - Negative: Better at B-type disambiguation
  - Large absolute value = asymmetric learning

Phenomenon Labels:
  - Temporal: A=High(V1), B=Low(V2)
  - Instrument: A=Instrument, B=Possession
  - Serial: A=Nested, B=Flat
    """)
    print("="*100)