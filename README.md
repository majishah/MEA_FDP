# Ridge & Lasso Regression — Diabetes Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/YOUR_COLAB_LINK_HERE)

Compare Standard Linear Regression, Ridge (L2), and Lasso (L1) regularization on the Diabetes dataset. Hyperparameters tuned using cross-validation.

## Objective

Demonstrate why regularization matters, how Ridge and Lasso differ, and how cross-validation selects the optimal regularization strength (alpha).

## Dataset

**Diabetes Dataset** (built into scikit-learn — no manual download required)

| Property | Details |
|----------|---------|
| Samples | 442 |
| Features | 10 (age, sex, bmi, bp, s1–s6) |
| Target | Quantitative measure of disease progression one year after baseline |
| Source | `sklearn.datasets.load_diabetes()` |
| Preprocessing | Features are already mean-centered and scaled by standard deviation |

## Project Structure

```
├── Ridge_Lasso_Regression_Diabetes.ipynb   # Complete Colab notebook
├── README.md                                # This file
```

## Methods Implemented

### 1. Standard Linear Regression (Baseline)

No regularization — minimizes only the MSE.

$$J(\theta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### 2. Ridge Regression (L2 Regularization)

Adds a penalty proportional to the **square** of coefficients. Shrinks all coefficients but never sets any to exactly zero.

$$J(\theta) = MSE + \alpha \sum_{j=1}^{p} \theta_j^2$$

### 3. Lasso Regression (L1 Regularization)

Adds a penalty proportional to the **absolute value** of coefficients. Can shrink coefficients to exactly zero, performing automatic feature selection.

$$J(\theta) = MSE + \alpha \sum_{j=1}^{p} |\theta_j|$$

### Comparison

| Aspect | Linear | Ridge (L2) | Lasso (L1) |
|--------|--------|-----------|-----------|
| Regularization | None | α × Σ(θ²) | α × Σ\|θ\| |
| Feature selection | No | No (keeps all) | Yes (drops irrelevant) |
| Multicollinearity | Struggles | Handles well | Handles well |
| When to use | Independent features | All features useful but correlated | Many irrelevant features suspected |

## Hyperparameter Tuning

Alpha (λ) controls regularization strength and is tuned using **5-fold cross-validation** via `RidgeCV` and `LassoCV`.

| Alpha too low | Alpha optimal | Alpha too high |
|---------------|--------------|----------------|
| Overfitting (model too complex) | Best generalization | Underfitting (model too simple) |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error — average of squared prediction errors |
| **RMSE** | Root MSE — interpretable in the same unit as the target |
| **R² Score** | Proportion of variance explained (0 to 1, higher is better) |
| **5-Fold CV R²** | Average R² across 5 cross-validation folds (most reliable metric) |

## Visualizations Included

1. **Feature scatter plots** — All 10 features vs target (2×5 grid)
2. **Feature correlation ranking** — Bar chart with target correlation
3. **Coefficient comparison** — Grouped bar chart (Linear vs Ridge vs Lasso)
4. **Ridge regularization path** — Coefficients vs alpha (all shrink, none reach zero)
5. **Lasso regularization path** — Coefficients vs alpha (some reach exactly zero)
6. **Cross-validation R² vs alpha** — For both Ridge and Lasso (finding the peak)
7. **Metrics bar charts** — MSE, R², CV R² side by side
8. **Residual plots** — All three models compared
9. **Actual vs predicted** — Scatter plots with perfect-prediction line

## How to Run

### Option A: Google Colab (Recommended)

1. Click the **"Open in Colab"** badge above
2. Or: open [Google Colab](https://colab.research.google.com/) → **File → Upload Notebook** → upload the `.ipynb` file
3. Click **Runtime → Run All**

No installations needed — all libraries are pre-installed in Colab.

### Option B: Local Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/ridge-lasso-regression-diabetes.git
cd ridge-lasso-regression-diabetes

# Install dependencies
pip install numpy pandas matplotlib scikit-learn

# Launch Jupyter
jupyter notebook Ridge_Lasso_Regression_Diabetes.ipynb
```

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Key Takeaways

- Regularization reduces overfitting by penalizing large coefficients.
- **Ridge** keeps all features but shrinks coefficients — best when all features are relevant.
- **Lasso** performs automatic feature selection by setting some coefficients to zero — best when many features are irrelevant.
- **Cross-validation** is essential for finding the optimal alpha (regularization strength).
- The Train–Test R² gap is smaller for regularized models, indicating better generalization.

## Human Learning Analogy

| ML Concept | Human Parallel |
|------------|---------------|
| Linear Regression | Studying everything equally — no strategy |
| Ridge (L2) | Balanced effort across all topics — nothing ignored, nothing over-studied |
| Lasso (L1) | Strategically dropping weak chapters, focusing on high-impact topics |
| Alpha (λ) | How strict your study discipline is |
| Cross-Validation | Taking multiple practice tests to find the best study strategy |
| Overfitting | Memorizing one textbook perfectly but failing different question papers |

## License

This project is for educational purposes. The Diabetes dataset is available via scikit-learn under a BSD license.
