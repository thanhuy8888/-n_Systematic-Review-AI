"""
DEPRECATED. This helper previously trained a TF-IDF + RandomForest model that
did NOT match the deployed screening model, and it printed *fabricated*
extraction / Cohen's-kappa numbers ("Gợi ý điền báo cáo: 88.5% ..."). Those
suggested values were never measured and must not appear in the thesis.

Use the single, leakage-free evaluator instead:

    python experiments/baselines/evaluate_screening_clean.py

It reports only genuine held-out metrics for the screening pipeline. Extraction
(Exact Match / Overlap F1) and inter-annotator agreement (Cohen's kappa) remain
unmeasured within the scope of the thesis and are listed as future work
(Section 5.3), consistent with Table 5.
"""
import sys

if __name__ == "__main__":
    sys.exit(
        "generate_report_metrics.py is deprecated (it produced fabricated "
        "numbers). Run: python experiments/baselines/evaluate_screening_clean.py"
    )
