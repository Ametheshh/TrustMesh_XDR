# TrustMesh XDR - Unified Reports & Evaluation Directory

This unified directory consolidates all empirical benchmark reports, ablation studies, triage results, and publication-grade evaluation charts for the TrustMesh XDR architecture.

---

## Index of Reports & Visualizations

| Report / Asset | Format | Description |
| :--- | :---: | :--- |
| [**MODEL_BENCHMARK_REPORT.md**](MODEL_BENCHMARK_REPORT.md) | Markdown | **Primary Benchmark Report**: Comprehensive evaluation comparing all 18 small models and the Unified Master Federated Model across **262,879 real network flows** (UNSW-NB15, CIC-IoT-2023, CIC-IoMT-2024). |
| [**AUDIT_REPORT.md**](AUDIT_REPORT.md) | Markdown | **Quality & Architecture Audit**: Full code audit adhering to senior-developer standards covering safety, scale, tests, and code economy. |
| [**models_benchmark_comparison.png**](models_benchmark_comparison.png) | Image (PNG) | **6-Panel Publication Chart**: High-resolution Matplotlib visual dashboard detailing Precision, Recall, F1, FPR, and Attack Category detection rates across all datasets. |
| [**multi_dataset_evaluation_report.json**](multi_dataset_evaluation_report.json) | JSON Data | Machine-readable metrics, per-class flow counts, and calibrated detection probabilities across all 262,879 flows. |
| [**centralized_vs_federated_summary.json**](centralized_vs_federated_summary.json) | JSON Data | Empirical benchmark comparing centralized pooled learning against privacy-preserving federated learning across seeds 42, 43, 44. |
| [**class_weighting_ablation_summary.json**](class_weighting_ablation_summary.json) | JSON Data | 4-way ablation study (Centralized Global-Weighted, Federated Local-Weighted, Centralized Unweighted, Federated Unweighted) measuring the impact of class imbalance compensation. |
| [**phase2_triage_results.md**](phase2_triage_results.md) | Markdown | Triage and correlation queue ranking report evaluating incident Recall@50 and Precision@50 on canonical connection streams. |

---

## How to Regenerate Reports

To re-run the 262,879-flow multi-dataset evaluation and regenerate `models_benchmark_comparison.png` and `multi_dataset_evaluation_report.json`:

```bash
cd /var/home/prashith/majorproj/TrustMesh_XDR
PYTHONPATH=. python3 scripts/compare_and_plot_models.py
```

To run the complete automated test suite (83 unit and integration tests):

```bash
cd /var/home/prashith/majorproj/TrustMesh_XDR
PYTHONPATH=. pytest
```
