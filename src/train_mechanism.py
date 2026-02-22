"""
Training script for CPP mechanism prediction model.

This script trains a model to classify CPPs by their uptake mechanism:
- Endocytosis (energy-dependent)
- Direct Translocation (energy-independent)

Usage:
    python src/train_mechanism.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import FeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_mechanism_data():
    """Load the mechanism dataset."""
    data_path = DATA_RAW / "cpp_mechanism_dataset.csv"
    df = pd.read_csv(data_path)
    logger.info(f"Dataset loaded: {len(df)} peptides")
    logger.info(f"  - Endocytosis: {(df['mechanism_label']==0).sum()}")
    logger.info(f"  - Translocation: {(df['mechanism_label']==1).sum()}")
    return df


def extract_features(df):
    """Extract features for all peptides."""
    logger.info("Extracting features...")
    extractor = FeatureExtractor()

    features_list = []
    for seq in df['sequence']:
        features = extractor.extract_all(seq)
        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    logger.info(f"  Features extracted: {features_df.shape[1]} features")

    return features_df


def train_and_evaluate_models(X, y, feature_names):
    """Train and evaluate multiple models."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(y_train)} ({(y_train==0).sum()} endo, {(y_train==1).sum()} trans)")
    logger.info(f"  Test: {len(y_test)} ({(y_test==0).sum()} endo, {(y_test==1).sum()} trans)")

    models = {
        'SVM-RBF': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
        ]),
        'SVM-Linear': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='linear', C=1.0, probability=True, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42))
        ]),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ])
    }

    results = {}
    best_model = None
    best_score = 0

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logger.info("\n" + "="*60)
    logger.info("MODEL EVALUATION")
    logger.info("="*60)

    for name, model in models.items():
        logger.info(f"\n{name}:")

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')

        logger.info(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        logger.info(f"  CV AUC-ROC:  {cv_auc.mean():.3f} (+/- {cv_auc.std()*2:.3f})")
        logger.info(f"  CV F1:       {cv_f1.mean():.3f} (+/- {cv_f1.std()*2:.3f})")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        test_acc = accuracy_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_proba)
        test_f1 = f1_score(y_test, y_pred)
        test_mcc = matthews_corrcoef(y_test, y_pred)

        logger.info(f"  Test Accuracy: {test_acc:.3f}")
        logger.info(f"  Test AUC-ROC:  {test_auc:.3f}")
        logger.info(f"  Test F1:       {test_f1:.3f}")
        logger.info(f"  Test MCC:      {test_mcc:.3f}")

        results[name] = {
            'cv_accuracy': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'cv_auc': float(cv_auc.mean()),
            'cv_f1': float(cv_f1.mean()),
            'test_accuracy': float(test_acc),
            'test_auc': float(test_auc),
            'test_f1': float(test_f1),
            'test_mcc': float(test_mcc)
        }

        if cv_auc.mean() > best_score:
            best_score = cv_auc.mean()
            best_model = (name, model)

    return results, best_model, X_test, y_test


def analyze_feature_importance(model, feature_names, output_dir):
    """Analyze feature importance."""
    clf = model.named_steps['clf']

    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        importances = np.abs(clf.coef_[0])
    else:
        logger.warning("This model does not support feature importance extraction")
        return None

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    top_features = importance_df.head(20)

    logger.info("\nTop 10 important features:")
    for i, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance')
    plt.title('Top 20 Features for Mechanism Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close()

    return importance_df


def plot_confusion_matrix(y_test, y_pred, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Endocytosis', 'Translocation'],
                yticklabels=['Endocytosis', 'Translocation'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Mechanism Prediction')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def main():
    """Main training pipeline."""

    logger.info("="*60)
    logger.info("CPP MECHANISM PREDICTOR - TRAINING")
    logger.info("="*60)

    df = load_mechanism_data()
    features_df = extract_features(df)

    X = features_df.values
    y = df['mechanism_label'].values
    feature_names = features_df.columns.tolist()

    results, (best_name, best_model), X_test, y_test = train_and_evaluate_models(
        X, y, feature_names
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"BEST MODEL: {best_name}")
    logger.info(f"{'='*60}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"training_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "mechanism_predictor.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"\nModel saved: {model_path}")

    importance_df = analyze_feature_importance(best_model, feature_names, output_dir)

    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, output_dir)

    logger.info("\nClassification report:")
    print(classification_report(y_test, y_pred,
                               target_names=['Endocytosis', 'Translocation']))

    results_summary = {
        'timestamp': timestamp,
        'dataset_size': len(df),
        'n_endocytosis': int((df['mechanism_label']==0).sum()),
        'n_translocation': int((df['mechanism_label']==1).sum()),
        'n_features': len(feature_names),
        'best_model': best_name,
        'all_results': results
    }

    with open(output_dir / "results_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")

    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)

    return results_summary


if __name__ == "__main__":
    main()
