# Linear Regression with One Variable — California Housing Dataset

A from-scratch implementation of Simple Linear Regression to predict California housing prices using a single feature (average number of rooms per dwelling).

## Objective

Implement and compare two fundamental approaches to linear regression:
- **Gradient Descent** (iterative optimization)
- **Normal Equation** (closed-form analytical solution)

Both are verified against Scikit-learn's built-in `LinearRegression`.

## Dataset

**California Housing Dataset** (built into scikit-learn — no manual download required)

| Property | Details |
|----------|---------|
| Samples | 20,640 |
| Features used | `AveRooms` (average rooms per dwelling) |
| Target | `MedHouseVal` (median house value in $100,000s) |
| Source | `sklearn.datasets.fetch_california_housing()` |

## Project Structure

```
├── Linear_Regression_California_Housing.ipynb   # Complete Colab notebook
├── README.md                                     # This file
```

## Methods Implemented

### 1. Gradient Descent (From Scratch)

Iteratively updates model parameters to minimize the cost function.

**Cost Function (MSE):**

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

**Parameter Update Rule:**

$$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

### 2. Normal Equation (Closed-Form)

Computes the optimal parameters in a single step — no iterations or learning rate needed.

$$\theta = (X^T X)^{-1} X^T y$$

### Comparison

| Aspect | Gradient Descent | Normal Equation |
|--------|-----------------|-----------------|
| Approach | Iterative | One-step analytical |
| Feature scaling | Required | Not required |
| Learning rate | Must be tuned | Not applicable |
| Large datasets | Scales well | Slow (matrix inversion is O(n³)) |
| Convergence | Approximate | Exact |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error — average of squared prediction errors |
| **RMSE** | Root MSE — interpretable in the same unit as the target |
| **R² Score** | Proportion of variance explained by the model (0 to 1) |

> **Note:** A low R² is expected here because we use only one feature. Housing prices depend on many factors (location, income, age of house, etc.), so a single-variable model captures limited variance.

## Visualizations Included

1. **Raw data scatter plot** — AveRooms vs House Value
2. **Feature distribution** — Histogram of AveRooms
3. **Gradient Descent convergence** — Cost vs iterations
4. **Fitted regression line** — All three methods compared
5. **Combined best-fit plot** — With equation and metrics overlay
6. **Residual analysis** — Residuals vs predicted + residual distribution
7. **Learning rate comparison** — Effect of different learning rates on convergence

## How to Run

### Option A: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Go to **File → Upload Notebook**
3. Upload `Linear_Regression_California_Housing.ipynb`
4. Click **Runtime → Run All**

No installations needed — all libraries are pre-installed in Colab.

### Option B: Local Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/linear-regression-california-housing.git
cd linear-regression-california-housing

# Install dependencies
pip install numpy pandas matplotlib scikit-learn

# Launch Jupyter
jupyter notebook Linear_Regression_California_Housing.ipynb
```

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Key Takeaways

- Both Gradient Descent and Normal Equation converge to the **same optimal solution**.
- Feature scaling is critical for Gradient Descent but unnecessary for the Normal Equation.
- Learning rate selection significantly impacts convergence speed and stability.
- A single feature provides limited predictive power — real-world models use multiple features.

## Human Learning Analogy

This project draws parallels between machine learning and human learning:

| ML Concept | Human Parallel |
|------------|---------------|
| Gradient Descent | Learning by trial and error — adjusting based on feedback |
| Normal Equation | Solving a problem directly using a known formula |
| Learning Rate | How big each correction step is (baby steps vs giant leaps) |
| Overfitting | Memorizing answers vs understanding concepts |
| Training/Test Split | Studying from textbooks, then facing an unseen exam |

## License

This project is for educational purposes. The California Housing dataset is available under a BSD license via scikit-learn.
