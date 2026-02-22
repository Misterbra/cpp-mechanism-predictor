"""
Proper nested cross-validation with feature selection INSIDE the CV loop.

This addresses the data leakage concern: feature selection must be done
on training data only, separately for each fold.

Usage:
    python src/proper_nested_cv.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from collections import Counter
import matplotlib.pyplot as plt
import json
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import FeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"


def sequence_identity(seq1, seq2):
    """Compute sequence identity between two peptides."""
    if len(seq1) != len(seq2):
        min_len = min(len(seq1), len(seq2))
        max_len = max(len(seq1), len(seq2))
        longer = seq1 if len(seq1) > len(seq2) else seq2
        shorter = seq2 if len(seq1) > len(seq2) else seq1
        best_identity = 0
        for i in range(max_len - min_len + 1):
            matches = sum(a == b for a, b in zip(shorter, longer[i:i+min_len]))
            identity = matches / max_len
            best_identity = max(best_identity, identity)
        return best_identity
    else:
        matches = sum(a == b for a, b in zip(seq1, seq2))
        return matches / len(seq1)


def filter_by_similarity(df, threshold=0.8):
    """Filter sequences by similarity."""
    sequences = df['sequence'].tolist()
    n = len(sequences)
    kept_indices = []
    removed_indices = set()

    for i in range(n):
        if i in removed_indices:
            continue
        kept_indices.append(i)
        for j in range(i + 1, n):
            if j not in removed_indices:
                sim = sequence_identity(sequences[i], sequences[j])
                if sim >= threshold:
                    removed_indices.add(j)

    return df.iloc[kept_indices].reset_index(drop=True)


def proper_nested_cv(X, y, feature_names, n_features_to_select=20, n_outer=5, n_bootstrap=1000):
    """
    Proper nested CV with feature selection INSIDE each fold.

    For each outer fold:
        1. Split into train/test
        2. On TRAIN ONLY: scale, select features with RFE
        3. Train model on selected features
        4. Evaluate on test (using same scaler and feature mask)
    """
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)

    all_y_true = []
    all_y_proba = []
    all_selected_features = []
    fold_results = []

    logger.info(f"\nProper Nested CV with feature selection INSIDE each fold")
    logger.info(f"n_features_to_select = {n_features_to_select}")
    logger.info("-" * 60)

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Step 1: Scale on TRAIN only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Step 2: Feature selection on TRAIN only
        estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=10)
        rfe.fit(X_train_scaled, y_train)

        # Get selected features for this fold
        selected_mask = rfe.support_
        selected_features = [f for f, s in zip(feature_names, selected_mask) if s]
        all_selected_features.append(selected_features)

        # Step 3: Train final model on selected features
        X_train_selected = X_train_scaled[:, selected_mask]
        X_test_selected = X_test_scaled[:, selected_mask]

        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(X_train_selected, y_train)

        # Step 4: Predict on test
        y_proba = model.predict_proba(X_test_selected)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        # Store results
        all_y_true.extend(y_test)
        all_y_proba.extend(y_proba)

        fold_auc = roc_auc_score(y_test, y_proba)
        fold_acc = accuracy_score(y_test, y_pred)
        fold_mcc = matthews_corrcoef(y_test, y_pred)

        fold_results.append({
            'fold': fold + 1,
            'auc': fold_auc,
            'accuracy': fold_acc,
            'mcc': fold_mcc,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'selected_features': selected_features
        })

        logger.info(f"Fold {fold+1}: AUC={fold_auc:.3f}, Acc={fold_acc:.3f}, MCC={fold_mcc:.3f}")
        logger.info(f"         Selected features: {selected_features[:5]}...")

    # Aggregate results
    all_y_true = np.array(all_y_true)
    all_y_proba = np.array(all_y_proba)
    all_y_pred = (all_y_proba >= 0.5).astype(int)

    # Bootstrap CI on aggregated predictions
    n_samples = len(all_y_true)
    bootstrap_aucs = []
    bootstrap_accs = []
    bootstrap_mccs = []

    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = all_y_true[indices]
        y_proba_boot = all_y_proba[indices]
        y_pred_boot = all_y_pred[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        bootstrap_aucs.append(roc_auc_score(y_true_boot, y_proba_boot))
        bootstrap_accs.append(accuracy_score(y_true_boot, y_pred_boot))
        bootstrap_mccs.append(matthews_corrcoef(y_true_boot, y_pred_boot))

    # Feature stability analysis
    feature_counts = Counter()
    for features in all_selected_features:
        feature_counts.update(features)

    # Features selected in ALL folds
    stable_features = [f for f, count in feature_counts.items() if count == n_outer]

    # Features selected in majority of folds
    majority_features = [f for f, count in feature_counts.items() if count >= n_outer // 2 + 1]

    results = {
        'auc': roc_auc_score(all_y_true, all_y_proba),
        'auc_ci_low': np.percentile(bootstrap_aucs, 2.5),
        'auc_ci_high': np.percentile(bootstrap_aucs, 97.5),
        'auc_std': np.std(bootstrap_aucs),
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'accuracy_ci_low': np.percentile(bootstrap_accs, 2.5),
        'accuracy_ci_high': np.percentile(bootstrap_accs, 97.5),
        'mcc': matthews_corrcoef(all_y_true, all_y_pred),
        'mcc_ci_low': np.percentile(bootstrap_mccs, 2.5),
        'mcc_ci_high': np.percentile(bootstrap_mccs, 97.5),
        'fold_results': fold_results,
        'feature_stability': {
            'stable_features': stable_features,
            'n_stable': len(stable_features),
            'majority_features': majority_features,
            'n_majority': len(majority_features),
            'feature_counts': dict(feature_counts.most_common())
        }
    }

    return results


def baseline_all_features(X, y, n_outer=5, n_bootstrap=1000):
    """Baseline: all features, no selection."""
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)

    all_y_true = []
    all_y_proba = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        all_y_true.extend(y_test)
        all_y_proba.extend(y_proba)

    all_y_true = np.array(all_y_true)
    all_y_proba = np.array(all_y_proba)
    all_y_pred = (all_y_proba >= 0.5).astype(int)

    # Bootstrap
    bootstrap_aucs = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(all_y_true), size=len(all_y_true), replace=True)
        if len(np.unique(all_y_true[indices])) < 2:
            continue
        bootstrap_aucs.append(roc_auc_score(all_y_true[indices], all_y_proba[indices]))

    return {
        'auc': roc_auc_score(all_y_true, all_y_proba),
        'auc_ci_low': np.percentile(bootstrap_aucs, 2.5),
        'auc_ci_high': np.percentile(bootstrap_aucs, 97.5),
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'mcc': matthews_corrcoef(all_y_true, all_y_pred)
    }


def main():
    """Run proper nested CV analysis."""

    logger.info("=" * 60)
    logger.info("PROPER NESTED CV - Feature Selection INSIDE Loop")
    logger.info("=" * 60)

    # Load and prepare data
    data_path = DATA_RAW / "cpp_mechanism_dataset.csv"
    df = pd.read_csv(data_path)
    logger.info(f"Original dataset: {len(df)} peptides")

    df_filtered = filter_by_similarity(df, threshold=0.8)
    logger.info(f"After 80% similarity filtering: {len(df_filtered)} peptides")

    # Extract features
    extractor = FeatureExtractor()
    features_list = [extractor.extract_all(seq) for seq in df_filtered['sequence']]
    features_df = pd.DataFrame(features_list)

    X = features_df.values
    y = df_filtered['mechanism_label'].values
    feature_names = features_df.columns.tolist()

    logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

    # Baseline: all features
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE: All 449 features (no selection)")
    logger.info("=" * 60)

    baseline = baseline_all_features(X, y)
    logger.info(f"AUC: {baseline['auc']:.3f} [{baseline['auc_ci_low']:.3f}-{baseline['auc_ci_high']:.3f}]")
    logger.info(f"Accuracy: {baseline['accuracy']:.3f}")
    logger.info(f"MCC: {baseline['mcc']:.3f}")

    # Test different numbers of features with PROPER nested CV
    logger.info("\n" + "=" * 60)
    logger.info("PROPER NESTED CV with varying n_features")
    logger.info("(Feature selection done INSIDE each fold)")
    logger.info("=" * 60)

    results_by_n = {}
    for n_feat in [10, 20, 30, 50]:
        logger.info(f"\n--- n_features = {n_feat} ---")
        results = proper_nested_cv(X, y, feature_names, n_features_to_select=n_feat)
        results_by_n[n_feat] = results

        logger.info(f"\nAggregated results:")
        logger.info(f"AUC: {results['auc']:.3f} [{results['auc_ci_low']:.3f}-{results['auc_ci_high']:.3f}]")
        logger.info(f"Accuracy: {results['accuracy']:.3f} [{results['accuracy_ci_low']:.3f}-{results['accuracy_ci_high']:.3f}]")
        logger.info(f"MCC: {results['mcc']:.3f} [{results['mcc_ci_low']:.3f}-{results['mcc_ci_high']:.3f}]")

        logger.info(f"\nFeature stability:")
        logger.info(f"  Features in ALL 5 folds: {results['feature_stability']['n_stable']}")
        logger.info(f"  Features in majority (>=3) folds: {results['feature_stability']['n_majority']}")

        if results['feature_stability']['stable_features']:
            logger.info(f"  Stable features: {results['feature_stability']['stable_features']}")

    # Find best n_features
    best_n = max(results_by_n.keys(), key=lambda n: results_by_n[n]['auc'])
    best_results = results_by_n[best_n]

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY (Proper Nested CV)")
    logger.info("=" * 60)

    logger.info(f"\n{'Method':<30} {'AUC':>8} {'95% CI':>20} {'Acc':>8} {'MCC':>8}")
    logger.info("-" * 80)
    logger.info(f"{'Baseline (449 features)':<30} {baseline['auc']:>8.3f} [{baseline['auc_ci_low']:.3f}-{baseline['auc_ci_high']:.3f}] {baseline['accuracy']:>8.3f} {baseline['mcc']:>8.3f}")

    for n_feat, results in sorted(results_by_n.items()):
        method = f"RFE nested (n={n_feat})"
        logger.info(f"{method:<30} {results['auc']:>8.3f} [{results['auc_ci_low']:.3f}-{results['auc_ci_high']:.3f}] {results['accuracy']:>8.3f} {results['mcc']:>8.3f}")

    logger.info(f"\nBest: n={best_n} features")

    # Check if results are still "too good"
    logger.info("\n" + "=" * 60)
    logger.info("SANITY CHECK")
    logger.info("=" * 60)

    if best_results['auc'] > 0.95:
        logger.warning("AUC > 0.95 is still very high. Consider:")
        logger.warning("1. Check for data leakage in the original dataset")
        logger.warning("2. Some peptides may have near-identical sequences")
        logger.warning("3. The task may be easier than expected")
    elif best_results['auc'] > 0.85:
        logger.info("AUC in reasonable range (0.85-0.95)")
    else:
        logger.info("AUC in modest range - consistent with biological complexity")

    # Feature stability assessment
    stability = best_results['feature_stability']
    stability_ratio = stability['n_stable'] / best_n
    logger.info(f"\nFeature stability: {stability['n_stable']}/{best_n} features stable across all folds ({stability_ratio:.1%})")

    if stability_ratio < 0.3:
        logger.warning("Low feature stability (<30%) - features are not robust")
    elif stability_ratio < 0.5:
        logger.info("Moderate feature stability (30-50%)")
    else:
        logger.info("Good feature stability (>50%)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"proper_nested_cv_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: AUC comparison
    ax1 = axes[0]
    methods = ['All (449)'] + [f'RFE (n={n})' for n in sorted(results_by_n.keys())]
    aucs = [baseline['auc']] + [results_by_n[n]['auc'] for n in sorted(results_by_n.keys())]
    ci_lows = [baseline['auc_ci_low']] + [results_by_n[n]['auc_ci_low'] for n in sorted(results_by_n.keys())]
    ci_highs = [baseline['auc_ci_high']] + [results_by_n[n]['auc_ci_high'] for n in sorted(results_by_n.keys())]

    errors = [[a - l for a, l in zip(aucs, ci_lows)],
              [h - a for a, h in zip(aucs, ci_highs)]]

    colors = ['#3498db'] + ['#2ecc71'] * len(results_by_n)
    bars = ax1.bar(methods, aucs, yerr=errors, capsize=5, color=colors, alpha=0.7)
    ax1.set_ylabel('AUC-ROC', fontsize=12)
    ax1.set_title('Proper Nested CV: AUC with 95% CI', fontsize=14)
    ax1.set_ylim(0.5, 1.0)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')

    for bar, auc in zip(bars, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=10)

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Feature stability
    ax2 = axes[1]
    n_feats = sorted(results_by_n.keys())
    stable_counts = [results_by_n[n]['feature_stability']['n_stable'] for n in n_feats]
    majority_counts = [results_by_n[n]['feature_stability']['n_majority'] for n in n_feats]

    x = np.arange(len(n_feats))
    width = 0.35

    ax2.bar(x - width/2, stable_counts, width, label='All 5 folds', color='#3498db', alpha=0.7)
    ax2.bar(x + width/2, majority_counts, width, label='Majority (>=3)', color='#2ecc71', alpha=0.7)
    ax2.plot(x, n_feats, 'ro--', label='Target n_features')

    ax2.set_xlabel('Target n_features', fontsize=12)
    ax2.set_ylabel('Number of stable features', fontsize=12)
    ax2.set_title('Feature Stability Across Folds', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(n_feats)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "proper_nested_cv_results.png", dpi=150)
    plt.close()

    # Save JSON
    summary = {
        'timestamp': timestamp,
        'method': 'proper_nested_cv',
        'description': 'Feature selection done INSIDE each CV fold to prevent data leakage',
        'n_samples': len(df_filtered),
        'baseline': baseline,
        'results_by_n_features': {
            str(n): {
                'auc': r['auc'],
                'auc_ci': [r['auc_ci_low'], r['auc_ci_high']],
                'accuracy': r['accuracy'],
                'mcc': r['mcc'],
                'feature_stability': r['feature_stability']
            }
            for n, r in results_by_n.items()
        },
        'best_n_features': best_n,
        'best_auc': best_results['auc']
    }

    with open(output_dir / "proper_nested_cv_results.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_dir}")

    return summary


if __name__ == "__main__":
    main()
