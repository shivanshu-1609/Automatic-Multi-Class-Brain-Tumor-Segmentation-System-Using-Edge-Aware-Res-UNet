# Quantitative Evaluation Metrics Comparison

## Foreground Mean Metrics (Tumor Classes)

| Metric | Simple_UNet | Edge-Aware UNet |
| :--- | :--- | :--- |
| **Dice Score (DSC)** | 0.7355 | 0.7181 |
| **Sensitivity** | 0.7563 | 0.7935 |
| **Specificity** | 0.9988 | 0.9984 |
| **IoU** | 0.6455 | 0.6187 |
| **Hausdorff Distance (HD95)** | N/A | 5.0615 |

## Per-Class Dice Scores

| Class | Simple_UNet | Edge-Aware UNet |
| :--- | :--- | :--- |
| **NCR/NET (Necrotic/Non-Enhancing Tumor Core)** | 0.6804 | 0.6642 |
| **ED (Peritumoral Edema)** | 0.7287 | 0.7157 |
| **ET (Enhancing Tumor)** | 0.7972 | 0.7743 |
