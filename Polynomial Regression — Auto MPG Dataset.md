# Polynomial Regression — Auto MPG Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/YOUR_COLAB_LINK_HERE)

Predict Miles Per Gallon (MPG) from engine displacement using polynomial regression of varying degrees, and compare with simple linear regression.

## Objective

Demonstrate why linear regression fails on non-linear data and how polynomial regression of different degrees captures the underlying curve, while also illustrating the **bias-variance tradeoff** (underfitting vs overfitting).

## Dataset

**Auto MPG Dataset** — loaded directly from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/auto+mpg) (no manual download needed).

| Property | Details |
|----------|---------|
| Samples | 398 (392 after removing missing values) |
| Feature used | `displacement` (engine size in cubic inches) |
| Target | `mpg` (miles per gallon) |
| Source | UCI ML Repository (loaded via URL in code) |

## Project Structure

```
├── Polynomial_Regression_Auto_MPG.ipynb   # Complete Colab notebook
├── README.md                               # This file
```

## Methods Implemented

### 1. Linear Regression (Degree 1 — Baseline)

Fits a straight line through the data.

$$\hat{y} = \theta_0 + \theta_1 \cdot x$$

### 2. Polynomial Regression (Degrees 2, 3, 4, 5)

Transforms the single feature into polynomial features, then applies linear regression on the expanded feature set.

$$\hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2 + \cdots + \theta_n x^n$$

Each model uses a **Pipeline**: `PolynomialFeatures → StandardScaler → LinearRegression`

### Why Polynomial Regression?

When the relationship between feature and target is curved (non-linear), a straight line will always underfit. Polynomial regression captures that curve by adding higher-order terms (x², x³, etc.) while still using linear regression under the hood.

### Comparison Summary

| Aspect | Linear (Deg 1) | Polynomial (Deg 2+) |
|--------|----------------|---------------------|
| Complexity | Simplest | Increases with degree |
| Curved data | Underfits | Captures the curve |
| Overfitting risk | Low | Increases with degree |
| Interpretability | High | Decreases with degree |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error — average of squared prediction errors |
| **RMSE** | Root MSE — error in the same unit as MPG |
| **R² Score** | Proportion of variance explained (0 to 1, higher is better) |

## Visualizations Included

1. **Raw data scatter plot** — Displacement vs MPG (shows the non-linear relationship)
2. **Distribution of MPG** — Histogram
3. **All polynomial fits on one plot** — Degrees 1–5 overlaid for direct comparison
4. **Individual degree-wise plots** — Each polynomial vs linear baseline (2×2 grid)
5. **Metrics bar charts** — MSE, RMSE, R² compared across all degrees
6. **Residual plots** — For each degree (pattern = underfitting, random = good fit)
7. **Overfitting detection plot** — Train R² vs Test R² with gap visualization
8. **Prediction demo** — Sample predictions comparing linear vs best polynomial

## How to Run

### Option A: Google Colab (Recommended)

1. Click the **"Open in Colab"** badge above
2. Or: open [Google Colab](https://colab.research.google.com/) → **File → Upload Notebook** → upload the `.ipynb` file
3. Click **Runtime → Run All**

No installations needed — all libraries are pre-installed in Colab.

### Option B: Local Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/polynomial-regression-auto-mpg.git
cd polynomial-regression-auto-mpg

# Install dependencies
pip install numpy pandas matplotlib scikit-learn

# Launch Jupyter
jupyter notebook Polynomial_Regression_Auto_MPG.ipynb
```

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Key Takeaways

- Linear regression **underfits** when the true relationship is curved.
- Degree 2 polynomial captures the non-linear pattern significantly better.
- Higher degrees (4, 5) offer diminishing returns and risk **overfitting**.
- The **Train R² vs Test R² gap** is a practical way to detect overfitting.
- Choosing the right degree is an example of the **bias-variance tradeoff**.

## Human Learning Analogy

| ML Concept | Human Parallel |
|------------|---------------|
| Linear (Degree 1) | Oversimplifying — "all big engines are bad on fuel" |
| Polynomial (Degree 2) | Understanding nuance — "the relationship is curved" |
| High Degree (5+) | Overthinking — memorizing every exception instead of the rule |
| Underfitting | A student who barely studied — misses the key pattern |
| Overfitting | A student who memorized the textbook but fails new questions |
| Bias-Variance Tradeoff | Balancing simplicity with accuracy |

## License

This project is for educational purposes. The Auto MPG dataset is publicly available from the UCI Machine Learning Repository.
