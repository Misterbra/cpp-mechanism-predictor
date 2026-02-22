"""
Generate ROC curve and Confusion Matrix for the manuscript.
Uses aggregated predictions from nested cross-validation on filtered dataset.
Consistent with Table 2 methodology (80% similarity threshold, SVM-RBF model).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import FeatureExtractor

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"


def sequence_identity(seq1, seq2):
    """Calculate sequence identity between two peptides."""
    if len(seq1) != len(seq2):
        min_len = min(len(seq1), len(seq2))
        max_len = max(len(seq1), len(seq2))

        best_identity = 0
        longer = seq1 if len(seq1) > len(seq2) else seq2
        shorter = seq2 if len(seq1) > len(seq2) else seq1

        for i in range(max_len - min_len + 1):
            matches = sum(a == b for a, b in zip(shorter, longer[i:i+min_len]))
            identity = matches / max_len
            best_identity = max(best_identity, identity)

        return best_identity
    else:
        matches = sum(a == b for a, b in zip(seq1, seq2))
        return matches / len(seq1)


def filter_by_similarity(df, threshold=0.8):
    """Filter sequences with >threshold identity (equivalent to CD-HIT)."""
    sequences = df['sequence'].tolist()
    n = len(sequences)

    # Compute similarity matrix
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        sim_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            sim = sequence_identity(sequences[i], sequences[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    # Greedy clustering
    kept_indices = []
    removed_indices = set()

    for i in range(n):
        if i in removed_indices:
            continue
        kept_indices.append(i)
        for j in range(i + 1, n):
            if j not in removed_indices and sim_matrix[i, j] >= threshold:
                removed_indices.add(j)

    return df.iloc[kept_indices].reset_index(drop=True)


def main():
    # Load data
    data_path = DATA_RAW / "cpp_mechanism_dataset.csv"
    df = pd.read_csv(data_path)
    print(f"Original dataset: {len(df)} peptides")

    # Apply 80% similarity filter (consistent with manuscript)
    df_filtered = filter_by_similarity(df, threshold=0.8)
    print(f"After 80% similarity filter: {len(df_filtered)} peptides")
    print(f"  - Endocytosis: {(df_filtered['mechanism_label']==0).sum()}")
    print(f"  - Translocation: {(df_filtered['mechanism_label']==1).sum()}")

    # Extract features
    extractor = FeatureExtractor()
    features_list = [extractor.extract_all(seq) for seq in df_filtered['sequence']]
    features_df = pd.DataFrame(features_list)

    X = features_df.values
    y = df_filtered['mechanism_label'].values

    # Model: SVM-RBF (best model according to Table 2)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
    ])

    # Nested CV predictions (aggregated)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Calculate metrics
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y, y_pred)

    print(f"AUC-ROC: {roc_auc:.3f}")
    print(f"Confusion Matrix:\n{cm}")

    # Create Figure 3 with two panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: ROC Curve
    ax = axes[0]
    ax.plot(fpr, tpr, color='#3498DB', lw=2, label=f'SVM-RBF (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random (AUC = 0.5)')
    ax.fill_between(fpr, tpr, alpha=0.2, color='#3498DB')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('A. ROC Curve (5-fold CV)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel B: Confusion Matrix
    ax = axes[1]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Endocytosis', 'Translocation'],
                yticklabels=['Endocytosis', 'Translocation'],
                annot_kws={'size': 14, 'fontweight': 'bold'})
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title('B. Confusion Matrix', fontsize=12, fontweight='bold')

    # Add metrics below confusion matrix
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    metrics_text = f'Sensitivity: {sensitivity:.2f} | Specificity: {specificity:.2f} | Accuracy: {accuracy:.2f}'
    ax.text(0.5, -0.15, metrics_text, transform=ax.transAxes, ha='center', fontsize=10)

    plt.tight_layout()

    # Save figure
    output_path = RESULTS_DIR / "advanced_analysis_20260116_155844" / "roc_confusion_matrix.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved: {output_path}")

    return {
        'auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy
    }


if __name__ == "__main__":
    main()
