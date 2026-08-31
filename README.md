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
