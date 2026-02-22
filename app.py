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
import io

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import FeatureExtractor, EISENBERG_HYDROPHOBICITY

st.set_page_config(
    page_title="CPP Mechanism Predictor",
    page_icon="🧬",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "models" / "mechanism_predictor.pkl"
DATA_PATH = Path(__file__).parent / "data" / "raw" / "cpp_mechanism_dataset.csv"
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


@st.cache_data
def load_dataset():
    """Load the reference dataset for comparisons."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


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
        'confidence': float(max(proba)),
        'features': features
    }


def compute_dataset_distributions(df, extractor):
    """Compute feature distributions for the reference dataset."""
    endo = df[df['mechanism_label'] == 0]
    trans = df[df['mechanism_label'] == 1]

    endo_feats = [extractor.extract_all(s) for s in endo['sequence']]
    trans_feats = [extractor.extract_all(s) for s in trans['sequence']]

    return pd.DataFrame(endo_feats), pd.DataFrame(trans_feats)


def render_feature_comparison(features, df_ref, extractor):
    """Render a visual comparison of key features vs dataset distributions."""
    import matplotlib.pyplot as plt

    key_features = {
        'PHY_basic_ratio': 'Basic residue ratio (R+K)',
        'PHY_hydrophobic_ratio': 'Hydrophobic ratio',
        'AMP_hydrophobic_moment': 'Hydrophobic moment',
        'AMP_amphipathicity_index': 'Amphipathicity index',
        'PHY_charge_mean': 'Mean charge',
        'PHY_hydrophobicity_mean': 'Mean hydrophobicity',
    }

    endo = df_ref[df_ref['mechanism_label'] == 0]
    trans = df_ref[df_ref['mechanism_label'] == 1]

    endo_vals = {k: [extractor.extract_all(s).get(k, 0) for s in endo['sequence']] for k in key_features}
    trans_vals = {k: [extractor.extract_all(s).get(k, 0) for s in trans['sequence']] for k in key_features}

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for i, (feat_key, feat_name) in enumerate(key_features.items()):
        ax = axes[i]
        user_val = features.get(feat_key, 0)

        ax.hist(endo_vals[feat_key], bins=15, alpha=0.5, color='#3498db', label='Endocytosis', density=True)
        ax.hist(trans_vals[feat_key], bins=15, alpha=0.5, color='#e74c3c', label='Translocation', density=True)
        ax.axvline(user_val, color='black', linewidth=2, linestyle='--', label='Your peptide')
        ax.set_title(feat_name, fontsize=10)
        ax.legend(fontsize=7)

    plt.tight_layout()
    return fig


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
    df_ref = load_dataset()

    if model is None:
        st.error("Model not found. Please train the model first using `python src/train_mechanism.py`")
        return

    # Tabs for single prediction vs batch
    tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])

    # ==================== SINGLE PREDICTION ====================
    with tab_single:
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

                    # Feature analysis
                    with st.expander("Sequence properties", expanded=True):
                        seq = prediction['sequence']
                        feats = prediction['features']

                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.write(f"**Length:** {len(seq)} aa")
                            basic = sum(1 for aa in seq if aa in 'RK')
                            st.write(f"**Basic (R+K):** {basic} ({100*basic/len(seq):.0f}%)")
                        with col_b:
                            hydrophobic = sum(1 for aa in seq if aa in 'AILMFVW')
                            st.write(f"**Hydrophobic:** {hydrophobic} ({100*hydrophobic/len(seq):.0f}%)")
                            st.write(f"**Net charge:** {feats.get('PHY_net_charge', 0):+.0f}")
                        with col_c:
                            st.write(f"**Hydrophobic moment:** {feats.get('AMP_hydrophobic_moment', 0):.3f}")
                            st.write(f"**Amphipathicity:** {feats.get('AMP_amphipathicity_index', 0):.3f}")

                    # Feature comparison with dataset distributions
                    if df_ref is not None:
                        with st.expander("Feature comparison with known CPPs"):
                            with st.spinner("Computing distributions..."):
                                fig = render_feature_comparison(prediction['features'], df_ref, extractor)
                                st.pyplot(fig)
                            st.caption("Dashed line = your peptide. Blue = endocytosis CPPs. Red = translocation CPPs.")

    # ==================== BATCH PREDICTION ====================
    with tab_batch:
        st.subheader("Batch Prediction")
        st.markdown("""
        Upload a CSV file with a column named `sequence` (one peptide per row),
        or paste multiple sequences (one per line).
        """)

        input_method = st.radio("Input method", ["Paste sequences", "Upload CSV"], horizontal=True)

        sequences_to_predict = []

        if input_method == "Paste sequences":
            text_input = st.text_area(
                "Paste sequences (one per line)",
                placeholder="GRKKRRQRRRPPQ\nGIGAVLKVLTTGLPALISWIKRKRQQ\nRQIKIWFQNRRMKWKK",
                height=150
            )
            if text_input:
                sequences_to_predict = [s.strip() for s in text_input.strip().split('\n') if s.strip()]

        else:
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded_file:
                try:
                    upload_df = pd.read_csv(uploaded_file)
                    if 'sequence' in upload_df.columns:
                        sequences_to_predict = upload_df['sequence'].dropna().tolist()
                        st.success(f"Loaded {len(sequences_to_predict)} sequences")
                    else:
                        st.error("CSV must have a column named 'sequence'")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        if sequences_to_predict and st.button("🔮 Predict All", type="primary", key="batch_predict"):
            results = []
            progress = st.progress(0)

            for i, seq in enumerate(sequences_to_predict):
                valid, cleaned = validate_sequence(seq)
                if valid:
                    pred = predict_mechanism(cleaned, model, extractor)
                    results.append({
                        'Sequence': pred['sequence'],
                        'Length': len(pred['sequence']),
                        'Prediction': pred['prediction'],
                        'P(Endocytosis)': round(pred['probability_endocytosis'], 3),
                        'P(Translocation)': round(pred['probability_translocation'], 3),
                        'Confidence': round(pred['confidence'], 3),
                    })
                else:
                    results.append({
                        'Sequence': seq,
                        'Length': len(seq),
                        'Prediction': f'ERROR: {cleaned}',
                        'P(Endocytosis)': None,
                        'P(Translocation)': None,
                        'Confidence': None,
                    })
                progress.progress((i + 1) / len(sequences_to_predict))

            results_df = pd.DataFrame(results)
            st.subheader(f"Results ({len(results)} sequences)")
            st.dataframe(results_df, use_container_width=True)

            # Summary stats
            valid_results = results_df[results_df['Confidence'].notna()]
            if len(valid_results) > 0:
                n_endo = (valid_results['Prediction'] == 'Endocytosis').sum()
                n_trans = (valid_results['Prediction'] == 'Translocation').sum()
                st.write(f"**Endocytosis:** {n_endo} | **Translocation:** {n_trans}")

            # Download button
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "📥 Download results as CSV",
                csv_buffer.getvalue(),
                "cppmechpred_results.csv",
                "text/csv"
            )

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    <p><strong>CPP Mechanism Predictor</strong> | Author: NB</p>
    <p>Model: SVM-RBF | AUC-ROC: 0.797 [95% CI: 0.714-0.878] | 465 features</p>
    <p>Dataset: 142 CPPs with experimentally validated mechanisms</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
