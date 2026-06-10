"""
DEPRECATED — kept only for backward compatibility with run_evaluation.bat and
older documentation.

The original version of this script inflated its reported metrics by
(a) applying SMOTE to the test set and (b) blending ~60% of the training rows
back into the evaluation set ("report blending"). Both practices cause
train/test leakage and were removed because they make the numbers
unreproducible and indefensible.

All evaluation now lives in the leakage-free script below, which also
regenerates the hybrid screening artifacts (hybrid_xgb_model.pkl,
tfidf_vectorizer.pkl) and the result figures.
"""
from evaluate_screening_clean import main

if __name__ == "__main__":
    print("[notice] evaluate_transformer.py is deprecated; running the "
          "leakage-free evaluate_screening_clean.py instead.\n")
    main()
