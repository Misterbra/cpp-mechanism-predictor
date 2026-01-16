"""
CPP Mechanism Predictor - Web Application

A Streamlit web application for predicting the uptake mechanism
of Cell-Penetrating Peptides (endocytosis vs direct translocation).

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import FeatureExtractor

st.set_page_config(
    page_title="CPP Mechanism Predictor",
    page_icon="🧬",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "models" / "mechanism_predictor.pkl"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


@st.cache_resource
def load_model():
    """Load the trained model."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_resource
def get_extractor():
    """Get the feature extractor."""
    return FeatureExtractor()


def validate_sequence(sequence: str) -> tuple:
    """Validate a peptide sequence."""
    sequence = sequence.upper().strip()

    if not sequence:
        return False, "Please enter a sequence."

    if len(sequence) < 5:
        return False, "Sequence too short (minimum 5 amino acids)."

    if len(sequence) > 50:
        return False, "Sequence too long (maximum 50 amino acids)."

    invalid_chars = set(sequence) - VALID_AA
    if invalid_chars:
        return False, f"Invalid characters: {', '.join(invalid_chars)}. Use only standard amino acids."

    return True, sequence


def predict_mechanism(sequence: str, model, extractor) -> dict:
    """Predict the uptake mechanism for a sequence."""
    features = extractor.extract_all(sequence)
    X = pd.DataFrame([features]).values

    proba = model.predict_proba(X)[0]
    prediction = model.predict(X)[0]

    return {
        'sequence': sequence,
        'prediction': 'Translocation' if prediction == 1 else 'Endocytosis',
        'probability_endocytosis': float(proba[0]),
        'probability_translocation': float(proba[1]),
        'confidence': float(max(proba))
    }


def main():
    """Main application."""

    st.title("🧬 CPP Mechanism Predictor")
    st.markdown("""
    **The first machine learning tool for predicting Cell-Penetrating Peptide uptake mechanism.**

    This tool predicts whether a CPP enters cells via:
    - **Endocytosis** (energy-dependent, vesicular uptake)
    - **Direct Translocation** (energy-independent, membrane penetration)
    """)

    st.divider()

    model = load_model()
    extractor = get_extractor()

    if model is None:
        st.error("Model not found. Please train the model first using `python src/train_mechanism.py`")
        return

    st.subheader("Enter a peptide sequence")

    sequence_input = st.text_input(
        "Sequence (5-50 amino acids)",
        placeholder="RRRRRRRR",
        help="Enter a peptide sequence using standard amino acid letters (A-Y)"
    )

    with st.expander("Example sequences"):
        examples = {
            "TAT (48-60)": "GRKKRRQRRRPPQ",
            "Penetratin": "RQIKIWFQNRRMKWKK",
            "R8 (octaarginine)": "RRRRRRRR",
            "Melittin": "GIGAVLKVLTTGLPALISWIKRKRQQ",
            "GALA": "WEAALAEALAEALAEHLAEALAEALEALAA",
            "Magainin 2": "GIGKFLHSAKKFGKAFVGEIMNS"
        }

        cols = st.columns(2)
        for i, (name, seq) in enumerate(examples.items()):
            with cols[i % 2]:
                if st.button(f"{name}", key=name):
                    st.session_state.example_seq = seq

    if 'example_seq' in st.session_state:
        sequence_input = st.session_state.example_seq
        del st.session_state.example_seq
        st.rerun()

    if st.button("🔮 Predict Mechanism", type="primary"):
        if sequence_input:
            valid, result = validate_sequence(sequence_input)

            if not valid:
                st.error(result)
            else:
                with st.spinner("Analyzing sequence..."):
                    prediction = predict_mechanism(result, model, extractor)

                st.divider()
                st.subheader("Prediction Result")

                if prediction['prediction'] == 'Endocytosis':
                    st.success(f"**Predicted mechanism: Endocytosis** (energy-dependent)")
                else:
                    st.info(f"**Predicted mechanism: Direct Translocation** (energy-independent)")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Endocytosis",
                        f"{prediction['probability_endocytosis']*100:.1f}%"
                    )

                with col2:
                    st.metric(
                        "Translocation",
                        f"{prediction['probability_translocation']*100:.1f}%"
                    )

                st.progress(prediction['confidence'])
                st.caption(f"Confidence: {prediction['confidence']*100:.1f}%")

                with st.expander("Sequence details"):
                    st.code(prediction['sequence'])
                    st.write(f"**Length:** {len(prediction['sequence'])} amino acids")

                    basic = sum(1 for aa in prediction['sequence'] if aa in 'RK')
                    hydrophobic = sum(1 for aa in prediction['sequence'] if aa in 'AILMFVW')
                    st.write(f"**Basic residues (R, K):** {basic} ({100*basic/len(prediction['sequence']):.1f}%)")
                    st.write(f"**Hydrophobic residues:** {hydrophobic} ({100*hydrophobic/len(prediction['sequence']):.1f}%)")

    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    <p><strong>CPP Mechanism Predictor</strong> | Author: NB</p>
    <p>Model: Logistic Regression | AUC-ROC: 0.795 [95% CI: 0.711-0.872]</p>
    <p>Dataset: 142 CPPs with experimentally validated mechanisms</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
