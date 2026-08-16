This document summarizes the experiments conducted while developing the model and the reasoning behind each decision. Full code and raw output are available in `final_model.ipynb`.

---

### Main Experiments

The following experiments shaped the final model configuration:

- **Training data size:** evaluated different sample sizes and observed a performance plateau.
- **Model comparison:** Logistic Regression, Decision Tree, Random Forest → **Random Forest** selected (recall 0.74 vs. 0.65–0.69).
- **Class imbalance:** `class_weight='balanced'` vs. manual weights vs. oversampling → **`class_weight='balanced'`** selected (best recall/F2 among tested approaches).
- **Feature engineering:** tested derived features from the 3 production fields → **`overdue_ratio = days_since_last_review / (past_reviews_count + 1)`** selected (improved precision/F1 without requiring new data).
- **Decision threshold:** swept 0.30–0.60 comparing F1/F1.5/F2 → **0.45** selected as a balance point between the F1-optimal (0.55) and F2-optimal (0.40) thresholds, favoring recall while limiting excess false positives.

---

### Additional Experiments

Additional experiments were conducted in an attempt to improve performance beyond the selected model.

| Experiment                           |           PR-AUC | Decision     |
| ------------------------------------ | ---------------: | ------------ |
| Baseline (`overdue_ratio`)           |            0.319 | Kept         |
| 50/50 Oversampling                   |            0.317 | Not selected |
| 50/50 SMOTE                          |            0.307 | Not selected |
| `overdue_ratio` + `log_days`         | Same as baseline | Not selected |
| Interaction features                 |            0.319 | Not selected |
| `min_samples_leaf` sweep             | Same as baseline | Not selected |
| `RandomizedSearchCV` (unconstrained) |            ~0.32 | Rejected     |
| Last-3:5 review history              |            0.333 | Future work  |

**Summary:** Most additional experiments matched or slightly underperformed the baseline. The unconstrained hyperparameter search produced higher recall, but at the cost of substantially lower accuracy and precision, making the resulting configuration impractical. The last-5 review history experiment showed the most promising improvement in PR-AUC, but requires additional review-history data that is not currently part of the API contract. It was therefore documented as future work rather than included in the shipped model.

---