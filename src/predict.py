"""
Prediction interface for CPP mechanism.

Usage:
    from src.predict import predict_mechanism
    result = predict_mechanism("RRRRRRRR")
"""

import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Union, List

from .features import FeatureExtractor


MODEL_PATH = Path(__file__).parent.parent / "models" / "mechanism_predictor.pkl"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def load_model():
    """Load the trained model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please train the model first using: python src/train_mechanism.py"
        )
    return joblib.load(MODEL_PATH)


def validate_sequence(sequence: str) -> str:
    """
    Validate and clean a peptide sequence.

    Args:
        sequence: Input peptide sequence

    Returns:
        Cleaned uppercase sequence

    Raises:
        ValueError: If sequence is invalid
    """
    sequence = sequence.upper().strip()

    if not sequence:
        raise ValueError("Empty sequence")

    if len(sequence) < 5:
        raise ValueError("Sequence too short (minimum 5 amino acids)")

    if len(sequence) > 50:
        raise ValueError("Sequence too long (maximum 50 amino acids)")

    invalid_chars = set(sequence) - VALID_AA
    if invalid_chars:
        raise ValueError(f"Invalid characters: {invalid_chars}")

    return sequence


def predict_mechanism(sequence: str) -> Dict[str, Union[str, float]]:
    """
    Predict the uptake mechanism for a CPP sequence.

    Args:
        sequence: Peptide sequence (5-50 amino acids)

    Returns:
        Dict with prediction results:
            - sequence: Input sequence
            - prediction: 'Endocytosis' or 'Translocation'
            - probability_endocytosis: Probability of endocytosis
            - probability_translocation: Probability of translocation
            - confidence: Confidence score (max probability)
    """
    # Validate
    sequence = validate_sequence(sequence)

    # Load model and extractor
    model = load_model()
    extractor = FeatureExtractor()

    # Extract features
    features = extractor.extract_all(sequence)
    X = pd.DataFrame([features]).values

    # Predict
    proba = model.predict_proba(X)[0]
    prediction = model.predict(X)[0]

    return {
        'sequence': sequence,
        'prediction': 'Translocation' if prediction == 1 else 'Endocytosis',
        'probability_endocytosis': float(proba[0]),
        'probability_translocation': float(proba[1]),
        'confidence': float(max(proba))
    }


def predict_batch(sequences: List[str]) -> pd.DataFrame:
    """
    Predict mechanisms for multiple sequences.

    Args:
        sequences: List of peptide sequences

    Returns:
        DataFrame with predictions for each sequence
    """
    results = []
    for seq in sequences:
        try:
            result = predict_mechanism(seq)
            result['error'] = None
        except Exception as e:
            result = {
                'sequence': seq,
                'prediction': None,
                'probability_endocytosis': None,
                'probability_translocation': None,
                'confidence': None,
                'error': str(e)
            }
        results.append(result)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    test_sequences = [
        "RRRRRRRR",      # R8 - expected: Endocytosis
        "GRKKRRQRRRPPQ", # TAT - expected: Endocytosis
        "GIGAVLKVLTTGLPALISWIKRKRQQ",  # Melittin - expected: Translocation
    ]

    print("CPP Mechanism Predictor")
    print("=" * 50)

    for seq in test_sequences:
        result = predict_mechanism(seq)
        print(f"\nSequence: {result['sequence']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
