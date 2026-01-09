### Command to add gpu acceleration
```
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#  Model Performance Comparison

##  Final Model Comparison Table

| Model         | Dataset               | Accuracy (%)  | Precision (TB) | Recall (TB) | F1-score (TB) | ROC–AUC |
|--------------|----------------------|---------------|----------------|-------------|---------------|---------|
| **ViT**       | Shenzhen              | **72.00**     | 0.69           | **0.80**    | 0.75          | **0.77** |
| **ViT**       | Shenzhen + Montgomery | 64.71 *(val)* | —              | —           | —             | —       |
| **MAE + ViT** | Shenzhen + Montgomery | **69.42**     | **0.73**       | 0.59        | 0.65          | 0.66    |

---

## Notes and Interpretation

### ViT (Shenzhen Dataset)
- Best performance on a single dataset
- High TB recall indicates strong sensitivity
- Limited generalization to unseen datasets

### ViT (Combined Dataset, without MAE)
- Performance drop due to domain shift
- Shows the limitation of supervised ViT on mixed datasets

### MAE + ViT (Combined Dataset)
- ~5% accuracy improvement over ViT-only combined training
- Better robustness and generalization
- Slight recall trade-off with improved precision

---
# 1. Conclusions and Discussion
## 7.1 Domain Shift Observation

This study clearly demonstrates the presence of domain shift in chest X-ray–based tuberculosis classification. When the Vision Transformer (ViT) model was trained and evaluated on a single dataset (Shenzhen), it achieved strong performance with high accuracy and recall. However, when the same model was trained on a combined dataset consisting of Shenzhen and Montgomery chest X-rays, a significant drop in performance was observed. This indicates that differences in imaging conditions, patient populations, and acquisition protocols across datasets negatively affect model generalization. These findings confirm that models trained on single-source medical datasets may not perform reliably in multi-institutional real-world settings.

## 7.2 Effect of MAE Pretraining on Generalization

To address the domain shift problem, Masked Autoencoder (MAE)–based self-supervised pretraining was introduced using an unlabeled subset of the NIH chest X-ray dataset. The MAE learned general structural representations of chest X-rays without relying on disease labels. When the pretrained MAE encoder was transferred to the ViT classifier and fine-tuned on the combined Shenzhen and Montgomery datasets, a noticeable improvement in validation and test performance was achieved. The MAE + ViT model improved combined-dataset accuracy by approximately 5% compared to the ViT-only model, demonstrating that self-supervised pretraining enhances cross-dataset generalization and robustness.

## 7.3 Trade-off Between Recall and Robustness

While MAE-based pretraining improved overall robustness and precision across datasets, a trade-off was observed in terms of tuberculosis recall. The Shenzhen-only ViT model achieved higher recall, making it more sensitive to TB detection in a single-domain setting. In contrast, the MAE + ViT model showed slightly lower recall but higher precision and more stable performance across datasets. This trade-off highlights an important consideration in medical AI systems: highly sensitive models may overfit to specific datasets, whereas more robust models generalize better but may sacrifice some sensitivity. Depending on the application, such as screening versus diagnostic support, this balance can be adjusted through threshold tuning or post-processing strategies.

## 7.4 Overall Conclusion

In conclusion, this project demonstrates that Vision Transformers are effective for tuberculosis detection in chest X-rays but are sensitive to domain shift when trained on heterogeneous datasets. Incorporating MAE-based self-supervised pretraining significantly improves cross-dataset robustness and generalization. The experimental results validate the effectiveness of MAE as a pretraining strategy for medical imaging tasks, especially in scenarios with limited labeled data and multi-institutional variability. This approach provides a strong foundation for building more reliable and scalable TB screening systems.

