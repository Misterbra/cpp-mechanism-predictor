"""
Test de l'hypothèse d'asymétrie spatiale des résidus.

Hypothèse : Les CPPs à translocation directe ont une distribution asymétrique
de leurs résidus hydrophobes (concentrés à une extrémité) par rapport aux
CPPs à endocytose.

Ce script teste si cette hypothèse est statistiquement significative.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt

# Échelle d'hydrophobicité de Kyte-Doolittle
HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

def compute_asymmetry(sequence):
    """
    Calcule l'asymétrie d'hydrophobicité entre les deux moitiés de la séquence.

    Returns:
        delta_hydro: hydrophobicité(N-terminal) - hydrophobicité(C-terminal)
        abs_delta: valeur absolue de l'asymétrie
    """
    seq = sequence.upper()
    mid = len(seq) // 2

    n_terminal = seq[:mid]
    c_terminal = seq[mid:]

    # Hydrophobicité moyenne de chaque moitié
    hydro_n = np.mean([HYDROPHOBICITY.get(aa, 0) for aa in n_terminal])
    hydro_c = np.mean([HYDROPHOBICITY.get(aa, 0) for aa in c_terminal])

    delta = hydro_n - hydro_c

    return delta, abs(delta)


def compute_charge_asymmetry(sequence):
    """
    Calcule l'asymétrie de charge entre les deux moitiés.
    """
    CHARGE = {'K': 1, 'R': 1, 'D': -1, 'E': -1, 'H': 0.1}

    seq = sequence.upper()
    mid = len(seq) // 2

    n_terminal = seq[:mid]
    c_terminal = seq[mid:]

    charge_n = sum(CHARGE.get(aa, 0) for aa in n_terminal)
    charge_c = sum(CHARGE.get(aa, 0) for aa in c_terminal)

    return charge_n - charge_c, abs(charge_n - charge_c)


def main():
    # Charger le dataset
    data_path = Path(__file__).parent.parent / "data" / "raw" / "cpp_mechanism_dataset.csv"
    df = pd.read_csv(data_path)

    print("=" * 60)
    print("TEST DE L'HYPOTHÈSE D'ASYMÉTRIE SPATIALE")
    print("=" * 60)
    print(f"\nDataset: {len(df)} peptides")
    print(f"  - Endocytose (0): {(df['mechanism_label'] == 0).sum()}")
    print(f"  - Translocation (1): {(df['mechanism_label'] == 1).sum()}")

    # Calculer l'asymétrie pour chaque peptide
    df['delta_hydro'], df['abs_delta_hydro'] = zip(*df['sequence'].apply(compute_asymmetry))
    df['delta_charge'], df['abs_delta_charge'] = zip(*df['sequence'].apply(compute_charge_asymmetry))

    # Séparer par mécanisme
    endo = df[df['mechanism_label'] == 0]
    trans = df[df['mechanism_label'] == 1]

    print("\n" + "=" * 60)
    print("RÉSULTATS - ASYMÉTRIE D'HYDROPHOBICITÉ")
    print("=" * 60)

    print(f"\nEndocytose (n={len(endo)}):")
    print(f"  Delta hydro moyen: {endo['delta_hydro'].mean():.3f} +/- {endo['delta_hydro'].std():.3f}")
    print(f"  |Delta| hydro moyen: {endo['abs_delta_hydro'].mean():.3f} +/- {endo['abs_delta_hydro'].std():.3f}")

    print(f"\nTranslocation (n={len(trans)}):")
    print(f"  Delta hydro moyen: {trans['delta_hydro'].mean():.3f} +/- {trans['delta_hydro'].std():.3f}")
    print(f"  |Delta| hydro moyen: {trans['abs_delta_hydro'].mean():.3f} +/- {trans['abs_delta_hydro'].std():.3f}")

    # Tests statistiques
    print("\n--- Tests statistiques ---")

    # Test t pour delta_hydro (direction de l'asymétrie)
    t_stat, p_value_delta = stats.ttest_ind(endo['delta_hydro'], trans['delta_hydro'])
    print(f"\nTest t (Delta hydro): t={t_stat:.3f}, p={p_value_delta:.4f}")

    # Test t pour abs_delta_hydro (magnitude de l'asymétrie)
    t_stat_abs, p_value_abs = stats.ttest_ind(endo['abs_delta_hydro'], trans['abs_delta_hydro'])
    print(f"Test t (|Delta| hydro): t={t_stat_abs:.3f}, p={p_value_abs:.4f}")

    # Test de Mann-Whitney (non-paramétrique)
    u_stat, p_value_mw = stats.mannwhitneyu(endo['abs_delta_hydro'], trans['abs_delta_hydro'], alternative='two-sided')
    print(f"Mann-Whitney (|Delta| hydro): U={u_stat:.1f}, p={p_value_mw:.4f}")

    print("\n" + "=" * 60)
    print("RÉSULTATS - ASYMÉTRIE DE CHARGE")
    print("=" * 60)

    print(f"\nEndocytose (n={len(endo)}):")
    print(f"  Delta charge moyen: {endo['delta_charge'].mean():.3f} +/- {endo['delta_charge'].std():.3f}")
    print(f"  |Delta| charge moyen: {endo['abs_delta_charge'].mean():.3f} +/- {endo['abs_delta_charge'].std():.3f}")

    print(f"\nTranslocation (n={len(trans)}):")
    print(f"  Delta charge moyen: {trans['delta_charge'].mean():.3f} +/- {trans['delta_charge'].std():.3f}")
    print(f"  |Delta| charge moyen: {trans['abs_delta_charge'].mean():.3f} +/- {trans['abs_delta_charge'].std():.3f}")

    # Tests pour la charge
    t_stat_charge, p_value_charge = stats.ttest_ind(endo['abs_delta_charge'], trans['abs_delta_charge'])
    print(f"\nTest t (|Delta| charge): t={t_stat_charge:.3f}, p={p_value_charge:.4f}")

    u_stat_charge, p_value_mw_charge = stats.mannwhitneyu(endo['abs_delta_charge'], trans['abs_delta_charge'], alternative='two-sided')
    print(f"Mann-Whitney (|Delta| charge): U={u_stat_charge:.1f}, p={p_value_mw_charge:.4f}")

    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    alpha = 0.05
    significant_hydro = p_value_abs < alpha or p_value_mw < alpha
    significant_charge = p_value_charge < alpha or p_value_mw_charge < alpha

    if significant_hydro:
        print(f"\n[OK] L'asymétrie d'HYDROPHOBICITÉ est SIGNIFICATIVE (p < {alpha})")
        print("  -> Cette feature pourrait améliorer le modèle")
    else:
        print(f"\n[X] L'asymétrie d'hydrophobicité n'est PAS significative (p >= {alpha})")
        print("  -> Cette feature n'apporterait probablement rien")

    if significant_charge:
        print(f"\n[OK] L'asymétrie de CHARGE est SIGNIFICATIVE (p < {alpha})")
        print("  -> Cette feature pourrait améliorer le modèle")
    else:
        print(f"\n[X] L'asymétrie de charge n'est PAS significative (p >= {alpha})")
        print("  -> Cette feature n'apporterait probablement rien")

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot hydrophobicité
    axes[0].boxplot([endo['abs_delta_hydro'], trans['abs_delta_hydro']],
                     labels=['Endocytose', 'Translocation'])
    axes[0].set_ylabel('|Delta Hydrophobicité|')
    axes[0].set_title(f'Asymétrie d\'hydrophobicité\n(p={p_value_abs:.4f})')

    # Boxplot charge
    axes[1].boxplot([endo['abs_delta_charge'], trans['abs_delta_charge']],
                     labels=['Endocytose', 'Translocation'])
    axes[1].set_ylabel('|Delta Charge|')
    axes[1].set_title(f'Asymétrie de charge\n(p={p_value_charge:.4f})')

    plt.tight_layout()

    # Sauvegarder
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "asymmetry_test.png", dpi=150)
    print(f"\nFigure sauvegardée: {output_dir / 'asymmetry_test.png'}")

    plt.show()

    return {
        'hydro_significant': significant_hydro,
        'charge_significant': significant_charge,
        'p_value_hydro': p_value_abs,
        'p_value_charge': p_value_charge
    }


if __name__ == "__main__":
    results = main()
