"""
Advanced analyses for CPP mechanism prediction model.

This script implements:
1. t-SNE visualization of peptides by mechanism
2. Classification error analysis
3. Comparison of classical features vs ESM-2 embeddings

Usage:
    python src/advanced_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import json
import warnings
warnings.filterwarnings('ignore')

# Import des features
import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import FeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# 1. t-SNE VISUALIZATION
# ============================================================

def create_tsne_visualization(X, y, sequences, names, output_dir):
    """
    Create t-SNE visualization of peptides colored by mechanism.
    """
    logger.info("Creating t-SNE visualization...")

    from sklearn.manifold import TSNE

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality reduction
    reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    X_2d = reducer.fit_transform(X_scaled)

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': X_2d[:, 0],
        'y': X_2d[:, 1],
        'mechanism': ['Endocytosis' if label == 0 else 'Translocation' for label in y],
        'name': names,
        'sequence': sequences,
        'length': [len(s) for s in sequences]
    })

    # Main figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: By mechanism
    colors = {'Endocytosis': '#E74C3C', 'Translocation': '#3498DB'}
    for mechanism, color in colors.items():
        mask = plot_df['mechanism'] == mechanism
        axes[0].scatter(
            plot_df.loc[mask, 'x'],
            plot_df.loc[mask, 'y'],
            c=color,
            label=mechanism,
            alpha=0.7,
            s=60,
            edgecolors='white',
            linewidth=0.5
        )

    axes[0].set_xlabel('t-SNE 1', fontsize=12)
    axes[0].set_ylabel('t-SNE 2', fontsize=12)
    axes[0].set_title('t-SNE - Peptides by Uptake Mechanism', fontsize=14)
    axes[0].legend(fontsize=11)

    # Plot 2: By length
    scatter = axes[1].scatter(
        plot_df['x'],
        plot_df['y'],
        c=plot_df['length'],
        cmap='viridis',
        alpha=0.7,
        s=60,
        edgecolors='white',
        linewidth=0.5
    )
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label('Length (AA)', fontsize=11)

    axes[1].set_xlabel('t-SNE 1', fontsize=12)
    axes[1].set_ylabel('t-SNE 2', fontsize=12)
    axes[1].set_title('t-SNE - Peptides by Length', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / "tsne_visualization.png", dpi=200)
    plt.close()

    # Figure with annotations for known peptides
    fig, ax = plt.subplots(figsize=(14, 10))

    for mechanism, color in colors.items():
        mask = plot_df['mechanism'] == mechanism
        ax.scatter(
            plot_df.loc[mask, 'x'],
            plot_df.loc[mask, 'y'],
            c=color,
            label=mechanism,
            alpha=0.6,
            s=80,
            edgecolors='white',
            linewidth=0.5
        )

    # Annotate famous peptides
    famous_peptides = ['TAT(48-60)', 'Penetratin', 'R8', 'R9', 'Transportan',
                       'GALA', 'Melittin', 'Magainin 2', 'LL-37', 'pHLIP']

    for _, row in plot_df.iterrows():
        if any(famous in row['name'] for famous in famous_peptides):
            ax.annotate(
                row['name'],
                (row['x'], row['y']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )

    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('t-SNE - CPP Peptides with Annotations', fontsize=14)
    ax.legend(fontsize=11, loc='best')

    plt.tight_layout()
    plt.savefig(output_dir / "tsne_annotated.png", dpi=200)
    plt.close()

    logger.info("  t-SNE visualizations saved")

    return plot_df


# ============================================================
# 2. ERROR ANALYSIS
# ============================================================

def analyze_errors(X, y, sequences, names, feature_names, output_dir):
    """
    Detailed analysis of classification errors.
    """
    logger.info("Analyzing classification errors...")

    # Use cross_val_predict to get predictions on the entire dataset
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Predictions and probabilities
    y_pred = cross_val_predict(model, X, y, cv=cv)
    y_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]

    # Identify errors
    errors_mask = y_pred != y

    # Create analysis DataFrame
    analysis_df = pd.DataFrame({
        'name': names,
        'sequence': sequences,
        'true_mechanism': ['Endocytosis' if label == 0 else 'Translocation' for label in y],
        'predicted_mechanism': ['Endocytosis' if pred == 0 else 'Translocation' for pred in y_pred],
        'probability_translocation': y_proba,
        'correct': ~errors_mask,
        'length': [len(s) for s in sequences]
    })

    # Extract key features for analysis
    extractor = FeatureExtractor()
    key_features = []
    for seq in sequences:
        feats = extractor.compute_physicochemical(seq)
        key_features.append({
            'charge': feats.get('PHY_net_charge', 0),
            'hydrophobicity': feats.get('PHY_hydrophobicity_mean', 0),
            'basic_ratio': feats.get('PHY_basic_ratio', 0),
            'hydrophobic_ratio': feats.get('PHY_hydrophobic_ratio', 0)
        })

    key_features_df = pd.DataFrame(key_features)
    analysis_df = pd.concat([analysis_df, key_features_df], axis=1)

    # Error statistics
    n_errors = errors_mask.sum()
    n_total = len(y)

    logger.info(f"  Total errors: {n_errors}/{n_total} ({100*n_errors/n_total:.1f}%)")

    # Errors by type
    false_endo = ((y == 1) & (y_pred == 0)).sum()  # Translocation classified as Endo
    false_trans = ((y == 0) & (y_pred == 1)).sum()  # Endo classified as Trans

    logger.info(f"  False Endocytosis (Trans->Endo): {false_endo}")
    logger.info(f"  False Translocation (Endo->Trans): {false_trans}")

    # Misclassified peptides
    errors_df = analysis_df[~analysis_df['correct']].copy()
    errors_df = errors_df.sort_values('probability_translocation')

    logger.info(f"\n  Misclassified peptides:")
    for _, row in errors_df.head(10).iterrows():
        logger.info(f"    {row['name']}: {row['true_mechanism']} -> {row['predicted_mechanism']} "
                   f"(p={row['probability_translocation']:.2f}, charge={row['charge']:.1f})")

    # Save complete analysis
    errors_df.to_csv(output_dir / "misclassified_peptides.csv", index=False)
    analysis_df.to_csv(output_dir / "full_classification_analysis.csv", index=False)

    # Error visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Probability distribution by true class
    ax = axes[0, 0]
    for label, name, color in [(0, 'Endocytosis', '#E74C3C'), (1, 'Translocation', '#3498DB')]:
        mask = y == label
        ax.hist(y_proba[mask], bins=20, alpha=0.6, label=name, color=color)
    ax.axvline(0.5, color='black', linestyle='--', label='Threshold')
    ax.set_xlabel('Probability (Translocation)')
    ax.set_ylabel('Number of peptides')
    ax.set_title('Probability Distribution by True Class')
    ax.legend()

    # 2. Errors by characteristics
    ax = axes[0, 1]
    correct_df = analysis_df[analysis_df['correct']]
    error_df = analysis_df[~analysis_df['correct']]

    ax.scatter(correct_df['charge'], correct_df['hydrophobicity'],
               c='green', alpha=0.5, label='Correct', s=50)
    ax.scatter(error_df['charge'], error_df['hydrophobicity'],
               c='red', alpha=0.8, label='Error', s=80, marker='x')
    ax.set_xlabel('Net charge')
    ax.set_ylabel('Mean hydrophobicity')
    ax.set_title('Errors in Charge/Hydrophobicity Space')
    ax.legend()

    # 3. Confusion matrix
    ax = axes[1, 0]
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Endocytosis', 'Translocation'],
                yticklabels=['Endocytosis', 'Translocation'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    # 4. Length of errors vs correct
    ax = axes[1, 1]
    ax.boxplot([correct_df['length'], error_df['length']], labels=['Correct', 'Error'])
    ax.set_ylabel('Length (AA)')
    ax.set_title('Peptide Length: Correct vs Errors')

    plt.tight_layout()
    plt.savefig(output_dir / "error_analysis.png", dpi=200)
    plt.close()

    # Error pattern analysis
    error_patterns = {
        'total_errors': int(n_errors),
        'error_rate': float(n_errors / n_total),
        'false_endocytosis': int(false_endo),
        'false_translocation': int(false_trans),
        'avg_charge_errors': float(error_df['charge'].mean()) if len(error_df) > 0 else 0,
        'avg_charge_correct': float(correct_df['charge'].mean()),
        'avg_hydrophobicity_errors': float(error_df['hydrophobicity'].mean()) if len(error_df) > 0 else 0,
        'avg_hydrophobicity_correct': float(correct_df['hydrophobicity'].mean()),
        'avg_length_errors': float(error_df['length'].mean()) if len(error_df) > 0 else 0,
        'avg_length_correct': float(correct_df['length'].mean())
    }

    with open(output_dir / "error_patterns.json", 'w') as f:
        json.dump(error_patterns, f, indent=2)

    logger.info("  Error analysis saved")

    return analysis_df, errors_df


# ============================================================
# 3. CLASSICAL FEATURES VS ESM-2 COMPARISON
# ============================================================

def get_esm2_embeddings(sequences):
    """
    Get ESM-2 embeddings for sequences.
    Uses esm2_t6_8M model (smallest) for speed.
    """
    try:
        import torch
        import esm
    except ImportError:
        logger.warning("ESM not installed.")
        return None

    logger.info("  Loading ESM-2 model...")

    # Load model (smallest for speed)
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # Prepare data
    data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]

    embeddings = []
    batch_size = 32

    logger.info(f"  Extracting embeddings for {len(sequences)} sequences...")

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)

            results = model(batch_tokens, repr_layers=[6])
            token_representations = results["representations"][6]

            # Moyenne sur les positions (exclure les tokens spéciaux)
            for j, (_, seq) in enumerate(batch):
                seq_len = len(seq)
                seq_embedding = token_representations[j, 1:seq_len+1].mean(0)
                embeddings.append(seq_embedding.numpy())

    return np.array(embeddings)


def compare_features_vs_esm2(X_classic, y, sequences, output_dir):
    """
    Compare performance of classical features vs ESM-2 embeddings.
    """
    logger.info("Comparing Classical Features vs ESM-2...")

    results = {}

    # 1. Classical features
    logger.info("  Evaluating classical features...")
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba_classic = cross_val_predict(model, X_classic, y, cv=cv, method='predict_proba')[:, 1]
    auc_classic = roc_auc_score(y, y_proba_classic)

    results['classic_features'] = {
        'n_features': X_classic.shape[1],
        'auc': float(auc_classic)
    }
    logger.info(f"    AUC Features Classiques: {auc_classic:.3f} ({X_classic.shape[1]} features)")

    # 2. ESM-2 embeddings
    X_esm = get_esm2_embeddings(sequences)

    if X_esm is not None:
        logger.info("  Evaluating ESM-2 embeddings...")
        y_proba_esm = cross_val_predict(model, X_esm, y, cv=cv, method='predict_proba')[:, 1]
        auc_esm = roc_auc_score(y, y_proba_esm)

        results['esm2_embeddings'] = {
            'n_features': X_esm.shape[1],
            'auc': float(auc_esm)
        }
        logger.info(f"    AUC ESM-2: {auc_esm:.3f} ({X_esm.shape[1]} features)")

        # 3. Combined
        logger.info("  Evaluating combined features...")
        X_combined = np.hstack([X_classic, X_esm])
        y_proba_combined = cross_val_predict(model, X_combined, y, cv=cv, method='predict_proba')[:, 1]
        auc_combined = roc_auc_score(y, y_proba_combined)

        results['combined'] = {
            'n_features': X_combined.shape[1],
            'auc': float(auc_combined)
        }
        logger.info(f"    AUC Combiné: {auc_combined:.3f} ({X_combined.shape[1]} features)")

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = ['Classical\nFeatures', 'ESM-2\nEmbeddings', 'Combined']
        aucs = [auc_classic, auc_esm, auc_combined]
        colors = ['#3498DB', '#E74C3C', '#2ECC71']

        bars = ax.bar(methods, aucs, color=colors, alpha=0.8, edgecolor='black')

        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title('Comparison: Classical Features vs ESM-2 vs Combined', fontsize=14)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')

        # Add values on bars
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{auc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / "features_comparison.png", dpi=200)
        plt.close()

    else:
        logger.warning("  ESM-2 not available, comparison limited to classical features")
        results['esm2_embeddings'] = None
        results['combined'] = None

    # Save results
    with open(output_dir / "features_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_without_esm2(X_classic, y, output_dir):
    """
    Simplified version without ESM-2 (if not installed).
    Compares different feature subsets.
    """
    logger.info("Feature comparison (without ESM-2)...")

    # On va comparer: toutes features vs physico-chimiques seules vs AAC seules
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    # Toutes les features
    y_proba_all = cross_val_predict(model, X_classic, y, cv=cv, method='predict_proba')[:, 1]
    auc_all = roc_auc_score(y, y_proba_all)
    results['all_features'] = {'auc': float(auc_all), 'n_features': X_classic.shape[1]}

    logger.info(f"  Toutes features: AUC = {auc_all:.3f}")

    # Sauvegarder
    with open(output_dir / "features_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    """Advanced analysis pipeline."""

    logger.info("=" * 60)
    logger.info("ADVANCED ANALYSIS - CPP MECHANISM MODEL")
    logger.info("=" * 60)

    # Create results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"advanced_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = DATA_RAW / "cpp_mechanism_dataset.csv"
    df = pd.read_csv(data_path)
    logger.info(f"\nDataset: {len(df)} peptides")

    # Extract features
    logger.info("Extracting features...")
    extractor = FeatureExtractor()
    features_list = [extractor.extract_all(seq) for seq in df['sequence']]
    features_df = pd.DataFrame(features_list)

    X = features_df.values
    y = df['mechanism_label'].values
    sequences = df['sequence'].tolist()
    names = df['name'].tolist()
    feature_names = features_df.columns.tolist()

    # 1. t-SNE Visualization
    logger.info("\n" + "=" * 60)
    logger.info("1. t-SNE VISUALIZATION")
    logger.info("=" * 60)
    tsne_df = create_tsne_visualization(X, y, sequences, names, output_dir)

    # 2. Error Analysis
    logger.info("\n" + "=" * 60)
    logger.info("2. ERROR ANALYSIS")
    logger.info("=" * 60)
    analysis_df, errors_df = analyze_errors(X, y, sequences, names, feature_names, output_dir)

    # 3. Features Comparison
    logger.info("\n" + "=" * 60)
    logger.info("3. FEATURE COMPARISON")
    logger.info("=" * 60)

    try:
        import torch
        import esm
        comparison_results = compare_features_vs_esm2(X, y, sequences, output_dir)
    except ImportError:
        logger.warning("ESM-2 not available, using simplified version")
        comparison_results = run_without_esm2(X, y, output_dir)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nResults saved in: {output_dir}")
    logger.info(f"  - tsne_visualization.png")
    logger.info(f"  - tsne_annotated.png")
    logger.info(f"  - error_analysis.png")
    logger.info(f"  - misclassified_peptides.csv")
    logger.info(f"  - features_comparison.json")

    return {
        'output_dir': str(output_dir),
        'n_peptides': len(df),
        'n_errors': len(errors_df),
        'comparison': comparison_results
    }


if __name__ == "__main__":
    main()
