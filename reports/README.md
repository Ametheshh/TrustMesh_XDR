# TrustMesh XDR — Presentation Slides & Benchmark Asset Guide

This directory contains the presentation assets and empirical benchmark charts for preparing presentation slides.

---

## 1. Quick Reference: Which Files to Refer To for Presentation Slides

If you or **Keerthan** are preparing presentation slides, refer to these **3 primary files**:

1. [**`docs/DEMONSTRATION_GUIDE.md`**](../docs/DEMONSTRATION_GUIDE.md): **The Complete Presentation Script & Speaker Notes**.
   - Contains slide-by-slide talking points, problem statement, math formulas, live terminal demo commands, and answers to evaluator questions.
2. [**`reports/MODEL_BENCHMARK_REPORT.md`**](MODEL_BENCHMARK_REPORT.md): **The Presentation Data & Statistics Source**.
   - Contains exact numerical tables, percentage detection rates across **262,879 real network flows**, attack breakdowns, and comparisons across all models.
3. [**`reports/slide1` to `slide6` PNG images**](#2-presentation-slide-image-mapping): **The Individual Slide Visualizations**.
   - High-resolution, presentation-proportioned (16:9) charts ready to drop directly into slide decks.

---

## 2. Presentation Slide Image Mapping

Each benchmark graphic is separated into an individual image with large fonts and bold callouts for slide readability:

| Slide # | Recommended Slide Title | Visual Asset Path | Key Bullet Points & Takeaways to Present |
| :---: | :--- | :--- | :--- |
| **Slide 1** | **Cross-Dataset Detection Performance** | [`reports/slide1_f1_cross_dataset_comparison.png`](slide1_f1_cross_dataset_comparison.png) | • Evaluated across **262,879 real network flows**.<br>• Compares Centralized Pooled vs FedAvg vs Unified Master Model.<br>• Master model achieves balanced high F1 score without pooling private raw packets. |
| **Slide 2** | **Modern IoT Attack Defense (CIC-IoT-2023)** | [`reports/slide2_iot_attack_detection.png`](slide2_iot_attack_detection.png) | • 134,231 flows evaluated.<br>• **100.0% Precision with 0.00% False Alarms** on 19,999 benign IoT devices at $t=0.50$.<br>• Catches **99.8% of DoS floods** and **99.8% of Mirai botnet traffic**. |
| **Slide 3** | **Healthcare IoMT & MQTT Defense (CIC-IoMT-2024)** | [`reports/slide3_healthcare_iomt_detection.png`](slide3_healthcare_iomt_detection.png) | • 46,316 flows across medical hospital sensors.<br>• Defends against MQTT-DoS connection floods, ARP cache poisoning, and OS reconnaissance.<br>• Maintains under **5% false alarm rate** on medical devices. |
| **Slide 4** | **Enterprise Campus Defense (UNSW-NB15)** | [`reports/slide4_enterprise_attack_detection.png`](slide4_enterprise_attack_detection.png) | • 82,332 flows of enterprise office traffic.<br>• **87.3% Precision** and **62.5% Recall** at calibrated threshold $0.15$.<br>• Catches **96.5% Generic exploits**, **90.3% Analysis probes**, and **87.1% Backdoors**. |
| **Slide 5** | **The Physics of Intrusion Detection (Learned Weights)** | [`reports/slide5_learned_feature_weights.png`](slide5_learned_feature_weights.png) | • Explains the mathematical physics learned by federated learning:<br>  - `packets_src_to_dst` (+1.16): Strongest attack predictor (surges in outbound packets).<br>  - `bytes_dst_to_src` (-0.43): High return data volume indicates normal downloads.<br>  - `duration` (-0.54): Stable long connections are benign. |
| **Slide 6** | **Precision vs False Alarm Rate Tradeoff** | [`reports/slide6_precision_vs_fpr_tradeoff.png`](slide6_precision_vs_fpr_tradeoff.png) | • Upper-left quadrant is ideal defense (high precision, low false alarm rate).<br>• **Centralized models fail**: Overfit to attack ratios and trigger up to 100% false alarms on modern IoT.<br>• **Unified Master Model wins**: Operates in the top-left ideal zone across both enterprise and IoT traffic. |
| **Backup** | **Full 6-Panel Benchmark Overview** | [`reports/models_benchmark_comparison.png`](models_benchmark_comparison.png) | • Comprehensive overview dashboard containing all 6 panels in a single composite graphic. |

---

## 3. How to Run the Live Demo on Slide Presentation Day

During the presentation, Keerthan can run this 1-line command in terminal to show the live federated model classifying live attack flows:

```bash
cd /var/home/prashith/majorproj/TrustMesh_XDR
python3 scripts/demo_unified_model.py
```
*(Executes in 0.05 seconds, displays clean colored terminal output showing 96%+ detection confidence on live attacks and 99.7% confidence on normal browsing).*
