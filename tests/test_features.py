"""Unit tests for feature extraction."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import FeatureExtractor
import numpy as np


def test_feature_count():
    """FeatureExtractor should produce exactly 465 features."""
    extractor = FeatureExtractor()
    features = extractor.extract_all("GRKKRRQRRRPPQ")
    assert len(features) == 465, f"Expected 465 features, got {len(features)}"


def test_aac_sums_to_one():
    """AAC frequencies should sum to ~1.0 for a valid sequence."""
    extractor = FeatureExtractor()
    features = extractor.extract_all("GRKKRRQRRRPPQ")
    aac_sum = sum(v for k, v in features.items() if k.startswith("AAC_"))
    assert abs(aac_sum - 1.0) < 1e-6, f"AAC sum should be ~1.0, got {aac_sum}"


def test_dpc_sums_to_one():
    """DPC frequencies should sum to ~1.0 for sequences >= 2 AA."""
    extractor = FeatureExtractor()
    features = extractor.extract_all("GRKKRRQRRRPPQ")
    dpc_sum = sum(v for k, v in features.items() if k.startswith("DPC_"))
    assert abs(dpc_sum - 1.0) < 1e-6, f"DPC sum should be ~1.0, got {dpc_sum}"


def test_length_feature():
    """Length feature should match actual sequence length."""
    extractor = FeatureExtractor()
    seq = "RRRRRRRR"
    features = extractor.extract_all(seq)
    assert features['length'] == 8


def test_hydrophobic_moment_amphipathic():
    """Amphipathic peptides should have higher hydrophobic moment than uniform cationic ones."""
    extractor = FeatureExtractor()
    # Melittin is amphipathic
    melittin_f = extractor.extract_all("GIGAVLKVLTTGLPALISWIKRKRQQ")
    # R8 is uniformly cationic
    r8_f = extractor.extract_all("RRRRRRRR")
    # Melittin should be more amphipathic
    assert melittin_f['AMP_amphipathicity_index'] > r8_f['AMP_amphipathicity_index']


def test_basic_ratio():
    """R8 (all arginines) should have basic ratio = 1.0."""
    extractor = FeatureExtractor()
    features = extractor.extract_all("RRRRRRRR")
    assert features['PHY_basic_ratio'] == 1.0


def test_positional_symmetry():
    """A symmetric sequence should have ~0 positional delta."""
    extractor = FeatureExtractor()
    features = extractor.extract_all("RRRRLLLLLLLLRRRR")
    # This is somewhat symmetric, so charge delta should be small
    assert abs(features['POS_charge_delta']) < 5


def test_predict_mechanism():
    """End-to-end prediction should return valid result."""
    import joblib
    model_path = Path(__file__).parent.parent / "models" / "mechanism_predictor.pkl"
    if not model_path.exists():
        return  # Skip if model not trained

    model = joblib.load(model_path)
    extractor = FeatureExtractor()

    features = extractor.extract_all("RRRRRRRR")
    X = np.array([list(features.values())])
    proba = model.predict_proba(X)[0]

    assert len(proba) == 2
    assert 0 <= proba[0] <= 1
    assert 0 <= proba[1] <= 1
    assert abs(sum(proba) - 1.0) < 1e-6


def test_batch_extraction():
    """Batch extraction should return correct number of rows."""
    extractor = FeatureExtractor()
    seqs = ["RRRRRRRR", "GIGAVLKVLTTGLPALISWIKRKRQQ", "RQIKIWFQNRRMKWKK"]
    df = extractor.extract_batch(seqs)
    assert len(df) == 3
    assert len(df.columns) == 465


if __name__ == "__main__":
    tests = [
        test_feature_count,
        test_aac_sums_to_one,
        test_dpc_sums_to_one,
        test_length_feature,
        test_hydrophobic_moment_amphipathic,
        test_basic_ratio,
        test_positional_symmetry,
        test_predict_mechanism,
        test_batch_extraction,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
