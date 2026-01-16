# CPP Mechanism Predictor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**The first machine learning model for predicting Cell-Penetrating Peptide (CPP) uptake mechanism.**

This tool predicts whether a CPP enters cells via **endocytosis** (energy-dependent) or **direct translocation** (energy-independent) based on its amino acid sequence.

---

## Why does this matter?

Cell-Penetrating Peptides are promising drug delivery vectors, but their therapeutic efficacy depends on *how* they enter cells:

| Mechanism | Process | Implication |
|-----------|---------|-------------|
| **Endocytosis** | Vesicular uptake | Cargo may be trapped in endosomes |
| **Direct Translocation** | Membrane penetration | Direct cytoplasmic access |

Existing tools (MLCPP 2.0, CPPred-RF) predict *uptake efficiency*, but **none predict the uptake mechanism**. This is the first.

---

## Quick Start

### Installation

```bash
git clone https://github.com/[username]/cpp-mechanism-predictor.git
cd cpp-mechanism-predictor
pip install -r requirements.txt
```

### Predict a sequence

```python
from src.predict import predict_mechanism

result = predict_mechanism("RRRRRRRR")
print(result)
# {'sequence': 'RRRRRRRR', 'prediction': 'Endocytosis', 'probability': 0.89}
```

### Web Application

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## Results

### Model Performance

Validated using nested 5-fold cross-validation with 1000 bootstrap iterations on a non-redundant dataset (80% sequence identity threshold).

| Model | AUC-ROC | 95% CI | Accuracy | MCC |
|-------|---------|--------|----------|-----|
| **SVM-RBF** | **0.795** | [0.711 - 0.872] | 72.1% | 0.447 |
| Logistic Regression | 0.786 | [0.698 - 0.866] | 73.9% | 0.474 |
| SVM-Linear | 0.753 | [0.659 - 0.840] | 73.0% | 0.456 |

### Comparison with ESM-2 Embeddings

| Features | AUC-ROC | # Features |
|----------|---------|------------|
| Classical (AAC, physicochemical) | 0.834 | 449 |
| ESM-2 embeddings | 0.819 | 320 |
| Combined | 0.830 | 769 |

Classical interpretable features perform as well as protein language model embeddings.

### Key Predictive Features

| Feature | Biological Interpretation |
|---------|--------------------------|
| Leucine content | Hydrophobic membrane interaction |
| Hydrophobicity | Direct translocation preference |
| Basic residue ratio (R, K) | Endocytosis via HSPG binding |
| Net charge | Electrostatic membrane interaction |

---

## Dataset

142 CPPs with experimentally validated uptake mechanisms, curated from peer-reviewed literature.

| Class | Count | Examples |
|-------|-------|----------|
| Endocytosis | 67 | TAT, Penetratin, R8, R9 |
| Translocation | 75 | Melittin, GALA, Magainin 2 |

After removing sequences with >80% identity: **111 non-redundant peptides**.

See [supplementary/Table_S1_peptide_references.csv](supplementary/Table_S1_peptide_references.csv) for the complete list with references.

---

## Project Structure

```
cpp-mechanism-predictor/
├── README.md
├── LICENSE
├── requirements.txt
├── app.py                      # Streamlit web application
├── src/
│   ├── features.py             # Feature extraction (449 features)
│   ├── train_mechanism.py      # Model training
│   ├── robust_validation.py    # Nested CV + bootstrap
│   ├── advanced_analysis.py    # t-SNE, error analysis, ESM-2 comparison
│   └── predict.py              # Prediction interface
├── data/
│   └── raw/
│       └── cpp_mechanism_dataset.csv
├── models/
│   └── mechanism_predictor.pkl
├── results/
│   └── [figures and analysis]
└── supplementary/
    └── Table_S1_peptide_references.csv
```

---

## Reproduce Results

```bash
# 1. Train the model
python src/train_mechanism.py

# 2. Run robust validation
python src/robust_validation.py

# 3. Run advanced analyses (t-SNE, error analysis, ESM-2)
python src/advanced_analysis.py
```

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{cpp_mechanism_predictor,
  author = {NB},
  title = {CPP Mechanism Predictor: First ML model for predicting cell-penetrating peptide uptake mechanism},
  year = {2025},
  url = {https://github.com/[username]/cpp-mechanism-predictor}
}
```

*Paper in preparation.*

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contact

**Author:** NB

For questions or collaborations, please open an issue on this repository.
