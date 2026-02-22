"""
Feature extraction for peptide sequences.

This module computes numerical representations of peptide sequences
for use as features in ML models.

Features implemented:
- AAC: Amino Acid Composition
- DPC: Dipeptide Composition
- Physicochemical properties
- PseAAC: Pseudo Amino Acid Composition
- Amphipathicity: Hydrophobic moment (Eisenberg et al. 1982)
- Positional features: N-terminal vs C-terminal property asymmetry

Usage:
    from src.features import FeatureExtractor
    extractor = FeatureExtractor()
    features = extractor.extract_all(sequence)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from collections import Counter
import math


# Standard 20 amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


# Physicochemical properties of amino acids
PHYSICOCHEMICAL_PROPERTIES = {
    # Hydrophobicity (Kyte-Doolittle scale)
    'hydrophobicity': {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    },
    # Charge at pH 7
    'charge': {
        'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
        'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,
        'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
        'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
    },
    # Polarity
    'polarity': {
        'A': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 0,
        'G': 0, 'H': 1, 'I': 0, 'K': 1, 'L': 0,
        'M': 0, 'N': 1, 'P': 0, 'Q': 1, 'R': 1,
        'S': 1, 'T': 1, 'V': 0, 'W': 0, 'Y': 1
    },
    # Volume (Angstrom^3)
    'volume': {
        'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
        'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
        'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
        'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
    },
    # Molecular weight (Da)
    'molecular_weight': {
        'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
        'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
        'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
        'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2
    },
    # Isoelectric point
    'pI': {
        'A': 6.0, 'C': 5.1, 'D': 2.8, 'E': 3.2, 'F': 5.5,
        'G': 6.0, 'H': 7.6, 'I': 6.0, 'K': 9.7, 'L': 6.0,
        'M': 5.7, 'N': 5.4, 'P': 6.3, 'Q': 5.7, 'R': 10.8,
        'S': 5.7, 'T': 5.6, 'V': 6.0, 'W': 5.9, 'Y': 5.7
    }
}


# Eisenberg consensus hydrophobicity scale (for hydrophobic moment calculation)
EISENBERG_HYDROPHOBICITY = {
    'A': 0.620, 'C': 0.290, 'D': -0.900, 'E': -0.740, 'F': 1.190,
    'G': 0.480, 'H': -0.400, 'I': 1.380, 'K': -1.500, 'L': 1.060,
    'M': 0.640, 'N': -0.780, 'P': 0.120, 'Q': -0.850, 'R': -2.530,
    'S': -0.180, 'T': -0.050, 'V': 1.080, 'W': 0.810, 'Y': 0.260
}


class FeatureExtractor:
    """
    Feature extractor for peptide sequences.

    Example:
        extractor = FeatureExtractor()
        features = extractor.extract_all("GRKKRRQRRRPPQ")
    """

    def __init__(self, pseaac_lambda: int = 5):
        """
        Initialize the extractor.

        Args:
            pseaac_lambda: Lambda parameter for PseAAC (number of correlations)
        """
        self.pseaac_lambda = pseaac_lambda
        self.amino_acids = AMINO_ACIDS
        self.properties = PHYSICOCHEMICAL_PROPERTIES
        self.dipeptides = [aa1 + aa2 for aa1 in AMINO_ACIDS for aa2 in AMINO_ACIDS]

    def compute_aac(self, sequence: str) -> Dict[str, float]:
        """
        Compute Amino Acid Composition (AAC).

        Args:
            sequence: Peptide sequence

        Returns:
            Dict with frequency of each amino acid
        """
        length = len(sequence)
        if length == 0:
            return {f"AAC_{aa}": 0.0 for aa in self.amino_acids}

        counts = Counter(sequence)
        return {f"AAC_{aa}": counts.get(aa, 0) / length for aa in self.amino_acids}

    def compute_dpc(self, sequence: str) -> Dict[str, float]:
        """
        Compute Dipeptide Composition (DPC).

        Args:
            sequence: Peptide sequence

        Returns:
            Dict with frequency of each dipeptide
        """
        length = len(sequence)
        if length < 2:
            return {f"DPC_{dp}": 0.0 for dp in self.dipeptides}

        dipeptide_counts = Counter()
        for i in range(length - 1):
            dp = sequence[i:i+2]
            if all(aa in self.amino_acids for aa in dp):
                dipeptide_counts[dp] += 1

        total = sum(dipeptide_counts.values())
        if total == 0:
            return {f"DPC_{dp}": 0.0 for dp in self.dipeptides}

        return {f"DPC_{dp}": dipeptide_counts.get(dp, 0) / total for dp in self.dipeptides}

    def compute_physicochemical(self, sequence: str) -> Dict[str, float]:
        """
        Compute global physicochemical properties.

        Args:
            sequence: Peptide sequence

        Returns:
            Dict with statistics for each property
        """
        features = {}

        for prop_name, prop_values in self.properties.items():
            values = [prop_values.get(aa, 0) for aa in sequence if aa in prop_values]

            if not values:
                features[f"PHY_{prop_name}_mean"] = 0.0
                features[f"PHY_{prop_name}_sum"] = 0.0
                features[f"PHY_{prop_name}_std"] = 0.0
                continue

            features[f"PHY_{prop_name}_mean"] = np.mean(values)
            features[f"PHY_{prop_name}_sum"] = np.sum(values)
            features[f"PHY_{prop_name}_std"] = np.std(values) if len(values) > 1 else 0.0

        # Special features for CPP
        charges = [self.properties['charge'].get(aa, 0) for aa in sequence]
        features['PHY_net_charge'] = sum(charges)
        features['PHY_positive_residues'] = sum(1 for c in charges if c > 0)
        features['PHY_negative_residues'] = sum(1 for c in charges if c < 0)

        hydrophobic = sum(1 for aa in sequence if self.properties['hydrophobicity'].get(aa, 0) > 0)
        features['PHY_hydrophobic_ratio'] = hydrophobic / len(sequence) if sequence else 0

        features['PHY_basic_ratio'] = (sequence.count('R') + sequence.count('K')) / len(sequence) if sequence else 0

        return features

    def compute_pseaac(self, sequence: str, lamda: Optional[int] = None) -> Dict[str, float]:
        """
        Compute Pseudo Amino Acid Composition (PseAAC).

        Args:
            sequence: Peptide sequence
            lamda: Number of correlations (default: self.pseaac_lambda)

        Returns:
            Dict with PseAAC features
        """
        if lamda is None:
            lamda = self.pseaac_lambda

        length = len(sequence)
        features = {}

        if length <= lamda:
            for i in range(lamda):
                features[f"PseAAC_corr_{i+1}"] = 0.0
            return features

        # Normalize properties
        normalized_props = {}
        for prop_name, prop_values in self.properties.items():
            values = list(prop_values.values())
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val > 0:
                normalized_props[prop_name] = {
                    aa: (v - mean_val) / std_val
                    for aa, v in prop_values.items()
                }
            else:
                normalized_props[prop_name] = prop_values.copy()

        # Compute correlations for each lag
        for lag in range(1, lamda + 1):
            corr_sum = 0
            count = 0

            for i in range(length - lag):
                aa1 = sequence[i]
                aa2 = sequence[i + lag]

                if aa1 in self.amino_acids and aa2 in self.amino_acids:
                    prop_diff = 0
                    n_props = 0
                    for prop_name, norm_values in normalized_props.items():
                        if aa1 in norm_values and aa2 in norm_values:
                            diff = (norm_values[aa1] - norm_values[aa2]) ** 2
                            prop_diff += diff
                            n_props += 1

                    if n_props > 0:
                        corr_sum += prop_diff / n_props
                        count += 1

            features[f"PseAAC_corr_{lag}"] = corr_sum / count if count > 0 else 0.0

        return features

    def compute_amphipathicity(self, sequence: str) -> Dict[str, float]:
        """
        Compute amphipathicity features including Eisenberg hydrophobic moment.

        The hydrophobic moment measures the amphipathicity of a peptide assuming
        an ideal alpha-helix (100 degrees per residue). High values indicate
        segregation of hydrophobic residues on one face of the helix, which is
        a key determinant of direct membrane translocation.

        Reference: Eisenberg et al. (1982) Nature 299:371-374
        """
        features = {}
        length = len(sequence)

        if length < 3:
            return {
                'AMP_hydrophobic_moment': 0.0,
                'AMP_hydrophobic_moment_normalized': 0.0,
                'AMP_mean_eisenberg_hydrophobicity': 0.0,
                'AMP_amphipathicity_index': 0.0,
            }

        # Hydrophobic moment for alpha-helix (100 degrees per residue)
        angle = 100.0  # degrees
        angle_rad = math.radians(angle)

        sum_sin = 0.0
        sum_cos = 0.0
        for i, aa in enumerate(sequence):
            h = EISENBERG_HYDROPHOBICITY.get(aa, 0.0)
            theta = i * angle_rad
            sum_sin += h * math.sin(theta)
            sum_cos += h * math.cos(theta)

        hydrophobic_moment = math.sqrt(sum_sin**2 + sum_cos**2) / length
        features['AMP_hydrophobic_moment'] = hydrophobic_moment

        # Mean Eisenberg hydrophobicity
        mean_h = np.mean([EISENBERG_HYDROPHOBICITY.get(aa, 0.0) for aa in sequence])
        features['AMP_mean_eisenberg_hydrophobicity'] = mean_h

        # Normalized hydrophobic moment (moment / mean absolute hydrophobicity)
        mean_abs_h = np.mean([abs(EISENBERG_HYDROPHOBICITY.get(aa, 0.0)) for aa in sequence])
        features['AMP_hydrophobic_moment_normalized'] = (
            hydrophobic_moment / mean_abs_h if mean_abs_h > 0 else 0.0
        )

        # Amphipathicity index: ratio of hydrophobic moment to mean hydrophobicity
        # High values = amphipathic (translocation), low = uniformly charged (endocytosis)
        features['AMP_amphipathicity_index'] = (
            hydrophobic_moment / (abs(mean_h) + 0.01)
        )

        return features

    def compute_positional(self, sequence: str) -> Dict[str, float]:
        """
        Compute positional features comparing N-terminal and C-terminal halves.

        These features capture the spatial distribution of physicochemical
        properties along the sequence, which can distinguish peptides that
        interact with membranes differently at each terminus.
        """
        features = {}
        length = len(sequence)

        if length < 4:
            return {
                'POS_hydro_nter': 0.0, 'POS_hydro_cter': 0.0,
                'POS_hydro_delta': 0.0, 'POS_hydro_abs_delta': 0.0,
                'POS_charge_nter': 0.0, 'POS_charge_cter': 0.0,
                'POS_charge_delta': 0.0, 'POS_charge_abs_delta': 0.0,
                'POS_hydrophobic_nter_ratio': 0.0,
                'POS_hydrophobic_cter_ratio': 0.0,
                'POS_basic_nter_ratio': 0.0,
                'POS_basic_cter_ratio': 0.0,
            }

        mid = length // 2
        n_ter = sequence[:mid]
        c_ter = sequence[mid:]

        hydro = self.properties['hydrophobicity']
        charge = self.properties['charge']

        # Hydrophobicity asymmetry
        hydro_n = np.mean([hydro.get(aa, 0) for aa in n_ter])
        hydro_c = np.mean([hydro.get(aa, 0) for aa in c_ter])
        features['POS_hydro_nter'] = hydro_n
        features['POS_hydro_cter'] = hydro_c
        features['POS_hydro_delta'] = hydro_n - hydro_c
        features['POS_hydro_abs_delta'] = abs(hydro_n - hydro_c)

        # Charge asymmetry
        charge_n = sum(charge.get(aa, 0) for aa in n_ter)
        charge_c = sum(charge.get(aa, 0) for aa in c_ter)
        features['POS_charge_nter'] = charge_n
        features['POS_charge_cter'] = charge_c
        features['POS_charge_delta'] = charge_n - charge_c
        features['POS_charge_abs_delta'] = abs(charge_n - charge_c)

        # Hydrophobic residue ratio per half
        features['POS_hydrophobic_nter_ratio'] = (
            sum(1 for aa in n_ter if hydro.get(aa, 0) > 0) / len(n_ter)
        )
        features['POS_hydrophobic_cter_ratio'] = (
            sum(1 for aa in c_ter if hydro.get(aa, 0) > 0) / len(c_ter)
        )

        # Basic residue ratio per half
        features['POS_basic_nter_ratio'] = (
            sum(1 for aa in n_ter if aa in 'RK') / len(n_ter)
        )
        features['POS_basic_cter_ratio'] = (
            sum(1 for aa in c_ter if aa in 'RK') / len(c_ter)
        )

        return features

    def extract_all(self, sequence: str) -> Dict[str, float]:
        """
        Extract all features from a sequence.

        Args:
            sequence: Peptide sequence

        Returns:
            Dict with all combined features
        """
        sequence = sequence.upper().strip()

        features = {}
        features['length'] = len(sequence)
        features.update(self.compute_aac(sequence))
        features.update(self.compute_dpc(sequence))
        features.update(self.compute_physicochemical(sequence))
        features.update(self.compute_pseaac(sequence))
        features.update(self.compute_amphipathicity(sequence))
        features.update(self.compute_positional(sequence))

        return features

    def extract_batch(self, sequences: List[str], show_progress: bool = False) -> pd.DataFrame:
        """
        Extract features for a list of sequences.

        Args:
            sequences: List of peptide sequences
            show_progress: Show progress bar

        Returns:
            DataFrame with one row per sequence
        """
        all_features = []
        for seq in sequences:
            features = self.extract_all(seq)
            all_features.append(features)

        return pd.DataFrame(all_features)


if __name__ == "__main__":
    # Test with TAT peptide
    tat = "GRKKRRQRRRPPQ"
    extractor = FeatureExtractor()

    print(f"Sequence: {tat}")
    print(f"Length: {len(tat)}")

    all_features = extractor.extract_all(tat)
    print(f"Total features: {len(all_features)}")
