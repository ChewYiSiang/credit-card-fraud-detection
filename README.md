# Credit Card Fraud Detection

End-to-end binary classification on a severely imbalanced dataset: 339 fraudulent transactions out of 20,000, a 1.7% base rate. Three model families are benchmarked against two imbalance-handling strategies, evaluated on PR-AUC rather than accuracy.

**Headline result: 0.941 ROC-AUC and 0.451 PR-AUC from a tuned Logistic Regression, beating both tree ensembles.**

## The problem

Accuracy is actively misleading here. A model that predicts "legitimate" for every transaction scores 98.3% accuracy and catches zero fraud. Worse, the baseline Random Forest at its default 0.5 threshold caught **0 of 68** fraud cases in the test set despite a respectable 0.905 ROC-AUC.

So the project optimises for PR-AUC, tunes the decision threshold explicitly, and treats the precision/recall tradeoff as the actual deliverable.

## Dataset

[Credit Card Fraud Detection 2026](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026) (Kaggle).

- 20,000 transactions, 26 features, no missing values
- 1.70% fraud rate, a realistic and severe imbalance
- Human-readable features, not PCA-anonymised, which is what makes the interpretability analysis below possible:
  - `merchant_risk_score`, `cvv_retry_count`, `velocity_score`, `ip_country_mismatch`
  - `billing_shipping_mismatch`, `distance_from_home_km`, `card_age_months`
  - `auth_method` (PIN, Biometric, 3D Secure, OTP, No Authentication)
  - `used_vpn`, `is_new_merchant`, `is_ai_generated_scam_attempt`

## Approach

1. **Baseline.** Random Forest, default settings, to establish a floor.
2. **Imbalance handling.** `class_weight='balanced'` compared against SMOTE oversampling, across every model family.
3. **Models.** Logistic Regression (L2), Random Forest, Gradient Boosting, XGBoost.
4. **Threshold tuning.** Decision threshold optimised per model to maximise F1 on the fraud class, rather than accepting the default 0.5 cutoff.
5. **Hyperparameter search.** Grid search over `C`, penalty (L1 / L2 / ElasticNet), and solver for the winning model.
6. **Interpretability.** Gini importance and logistic regression odds ratios, cross-checked against each other rather than trusting either alone.

## Results

| Model | ROC-AUC | PR-AUC | Recall (tuned) | Precision (tuned) |
|---|---|---|---|---|
| Random Forest (baseline) | 0.905 | 0.182 | 0.412 | 0.212 |
| **Logistic Regression (tuned)** | **0.941** | **0.451** | 0.515 | 0.422 |
| Logistic Regression (SMOTE) | 0.921 | 0.392 | 0.485 | 0.440 |
| Gradient Boosting | 0.923 | 0.338 | 0.324 | 0.512 |
| XGBoost | 0.921 | 0.271 | 0.485 | 0.266 |
| Random Forest (SMOTE) | 0.889 | 0.171 | 0.265 | 0.273 |

Three findings worth stating plainly:

- **Class weighting beat SMOTE on every single model.** With only 339 minority samples, SMOTE's synthetic interpolation generates unrealistic fraud examples, because there simply isn't enough local structure in the minority class to interpolate between sensibly.
- **Threshold tuning mattered more than model choice.** The baseline caught zero fraud at 0.5. The same family of decision, made explicitly, is the difference between a useless model and a deployable one.
- **The linear model won.** Fraud risk in this dataset is largely additive, so Logistic Regression outperformed every nonlinear ensemble tested. It also happens to be the most interpretable option, which is not usually a tradeoff you get for free.

## Top fraud signals

Consistent across both Gini importance and odds ratios:

| Feature | Odds ratio | Interpretation |
|---|---|---|
| `merchant_risk_score` | 1.71 | Strongest single predictor |
| `cvv_retry_count` | 1.70 | Repeated CVV failures indicate card testing |
| `velocity_score` | 1.55 | Rapid transaction velocity indicates account takeover |
| `ip_country_mismatch` | 1.49 | Geographic inconsistency flags compromise |
| `auth_method_No Authentication` | 1.47 | Weak authentication raises risk sharply |
| `auth_method_PIN` / `Biometric` | 0.85 | Strong authentication lowers fraud odds, as expected |

The last row is a useful sanity check. A model that did *not* find strong authentication protective would be suspect regardless of its AUC.

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), XGBoost, matplotlib, joblib.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook credit-card-fraud-detection.ipynb
```

The notebook reads the dataset from a Kaggle input path. Adjust the `pd.read_csv` path in the loading cell if running locally.

## What I would do differently

- **Pick the threshold from business cost, not F1.** F1 weights false positives and false negatives equally, which no fraud team actually wants. The right approach is a cost matrix (cost of a missed fraud versus cost of a blocked legitimate customer) and a threshold chosen to minimise expected cost. This is the single biggest gap between this notebook and a production system.
- **Split by time, not at random.** Fraud patterns drift. A random split lets the model see the future, so the reported numbers are optimistic relative to how it would behave deployed against next month's transactions.
- **Engineer velocity features properly.** `velocity_score` arrives precomputed. Rolling transaction counts over multiple windows (1h, 24h, 7d) would likely carry more signal than a single opaque score.
- **Get the notebook out of the notebook.** The pipeline is reproducible by hand but not automated. Extracting it into modules with a proper train/evaluate entrypoint would make the results reproducible by someone other than me.
