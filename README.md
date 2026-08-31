# Berry Picking Across Data Landscapes: Understanding Performance Sensitivity in Collaborative Filtering

[![Conference](https://img.shields.io/badge/Accept-Syndaite%202026%20%40%20ECML--PKDD-blue.svg)](https://ecmlpkdd.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official repository for the paper **"Berry Picking Across Data Landscapes: Understanding Performance Sensitivity in Collaborative Filtering"** accepted at the **Syndaite Workshop (ECML-PKDD 2026)**.

---

## Abstract

Understanding how collaborative filtering (CF) algorithms respond to variations in dataset characteristics remains an open challenge, particularly in sparse, domain-constrained settings. In this paper, we propose using **dataset morphing** to create semi-synthetic variants (*datasetoids*) between real recipe-rating datasets. By tracking performance along morphing trajectories together with extracted **metafeatures** and **landmarkers**, we demonstrate how comparative algorithmic advantage shifts continuously across dataset structures. Our results highlight the risk of *dataset selection bias* ("berry picking") in offline evaluations when relying on narrow, homogeneous datasets.

---

## Repository Structure

```text
.
├── extract_metafeatures.py  # Script for extracting metafeatures (Strategies A through E)
├── landmarkers.py           # Script for defining models and landmarking evaluation
├── morphing.py              # Script for executing dataset morphing via progressive row swapping
├── experiment_pipeline.ipynb # Master Jupyter Notebook running the entire experimental flow
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation
```
---

## Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/your-username/berry-picking-cf.git](https://github.com/your-username/berry-picking-cf.git)
cd berry-picking-cf

```


2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```

---

## How to Run the Experiments

The easiest way to reproduce the full experimental flow is through the provided Jupyter Notebook:

```bash
jupyter notebook experiment_pipeline.ipynb

```

The pipeline executes the three core experimental steps:

### Step 1: Dataset Morphing (`morphing.py`)

Generates intermediate synthetic datasets (*datasetoids*) between source and target datasets by progressively swapping rows (10% step replacement) across random seeds:

```python
from morphing import generate_morphed_datasets

generate_morphed_datasets(
    dataset_a_path="data/df_source.csv",
    dataset_b_path="data/df_target.csv",
    output_folder="output/morphed_datasets",
    swap_ratio=0.1,
    seed=42
)

```

### Step 2: Metafeature Extraction (`extract_metafeatures.py`)

Extracts statistical, structural, and interaction-based descriptors across strategies (`A`, `B`, `C`, `D`, `E`):

```python
from extract_metafeatures import process_metafeatures

process_metafeatures(
    input_folder="output/morphed_datasets",
    output_folder="output/metafeatures",
    strategy="A"  # Can loop through 'A', 'B', 'C', 'D', 'E'
)

```

### Step 3: Landmarker Evaluation (`landmarkers.py`)

Evaluates baseline and neural recommendation models (Most Pop, GMF, MLP, NeuMF, VAECF) on sampled versions of each datasetoid using ranking and evaluation metrics (AUC, NDCG@10, Precision@10, Recall@10):

```python
from landmarkers import extract_landmarkers_from_folder

extract_landmarkers_from_folder(
    datasets_dir="output/morphed_datasets",
    output_dir="output/landmarkers"
)

```

---

## Datasets

The study uses three recipe-rating datasets sampled from Food.com.
> ⏳ **Note:** The preprocessed datasets are currently being prepared and will be made available in this repository soon. In the meantime, you can place your own source and target CSV datasets formatted with `user_id`, `recipe_id`, and `rating` columns inside a `data/` folder to run the morphing pipeline.

---

## Citation

If you use this code or methodology in your research, please cite our paper:

```bibtex
@inproceedings{silva2026berry,
  title={Berry Picking Across Data Landscapes: Understanding Performance Sensitivity in Collaborative Filtering},
  author={Silva, Beatriz and Kokkinogenis, Zafeiris and Santos, Mois{\'e}s and Nunes, Francisco and Soares, Carlos},
  booktitle={Syndaite Workshop, ECML-PKDD},
  year={2026}
}

```

---

## Contact
* **LinkedIn**: [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)

```
