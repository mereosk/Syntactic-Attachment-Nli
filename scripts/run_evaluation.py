"""
Standalone script to run full evaluation.
Usage: python scripts/run_evaluation.py
"""


from src.config import MODELS_TO_TEST
from src.utils import load_data
from src.analysis import analyze_model_by_phenomenon
from src.visualization import (
    create_comprehensive_visualizations,
    create_summary_table,
    print_interpretation_guide
)


def main():
    """Run complete model evaluation."""
    
    print("="*100)
    print("STARTING NLI MODEL EVALUATION")
    print("="*100)
    
    # Load data
    data = load_data()
    
    print(f"\nLoaded {len(data)} examples")
    
    # Analyze all models
    all_model_results = {}
    
    for model_name in MODELS_TO_TEST:
        try:
            results = analyze_model_by_phenomenon(model_name, data)
            all_model_results[model_name] = results
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_model_results:
        print("No models successfully analyzed. Exiting.")
        return
    
    # Generate outputs
    print("\nGenerating results...")
    summary_df = create_summary_table(all_model_results, MODELS_TO_TEST)
    create_comprehensive_visualizations(all_model_results, MODELS_TO_TEST)
    print_interpretation_guide()
    
    print("\n" + "="*100)
    print("EVALUATION COMPLETE!")
    print("="*100)
    print("\nOutputs saved to:")
    print("  - phenomenon_comparison_results.csv")
    print("  - main_phenomenon_comparison.png")
    print("  - temporal_detailed_analysis.png")
    print("  - instrument_detailed_analysis.png")
    print("  - serial_detailed_analysis.png")


if __name__ == "__main__":
    main()
