# house-price-ml# 🏠 House Price Prediction — Machine Learning Project

## Overview

This project implements an **end-to-end machine learning regression pipeline** to predict California house prices based on socioeconomic and geographic features.
It demonstrates the **full ML lifecycle**: data loading, preprocessing, model training, evaluation, persistence, and reuse (inference).

The goal of this project is to build a **strong baseline regression model** and show a clean, reproducible ML workflow using Python and scikit-learn.

---

## Dataset

- **California Housing Dataset** (via `sklearn.datasets`)
- 20,640 samples
- 8 numerical features
- Target: `MedHouseVal` (median house value, in units of $100,000)

### Features

- `MedInc` – Median income
- `HouseAge` – Median house age
- `AveRooms` – Average number of rooms
- `AveBedrms` – Average number of bedrooms
- `Population` – Population in the area
- `AveOccup` – Average household occupancy
- `Latitude` – Latitude
- `Longitude` – Longitude

---

## Project Structure

```
house-price-ml/
├── notebook/
│   ├── house_price_prediction.ipynb   # Training & evaluation
│   └── reuse_model.ipynb               # Model reuse (inference)
├── model/
│   ├── linear_regression_model.pkl
│   └── house_price_pipeline.pkl        # Model + scaler
├── requirements.txt
├── README.md
└── venv/
```

---

## Machine Learning Pipeline

### 1. Data Preparation

- Loaded dataset using `fetch_california_housing`
- Converted to Pandas DataFrame
- Verified data quality (no missing values, all numeric)

### 2. Feature Engineering

- Separated features (`X`) and target (`y`)
- Applied **StandardScaler** to normalize feature scales

### 3. Model Training

- Algorithm: **Linear Regression**
- Train/Test split: 80% / 20%
- Trained only on scaled training data to avoid data leakage

### 4. Evaluation

Metrics used:

- **RMSE** (Root Mean Squared Error): ~0.75
- **MAE** (Mean Absolute Error): ~0.53

Interpretation:

- Average prediction error ≈ **$53,000**
- Provides a solid **baseline model** for future improvements

---

## Model Persistence & Reuse

The trained model and preprocessing scaler were saved using `pickle`:

```python
{
  "model": trained_linear_regression_model,
  "scaler": fitted_standard_scaler
}
```

### Reuse Workflow (Inference)

```
New data → scaler.transform → model.predict
```

Predictions can be generated for **new, unseen house data** using the saved artifacts.

---

## Example Prediction

```
Predicted House Value: ~2.44
≈ $244,000
```

---

## Technologies Used

- Python 3.11
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Jupyter Notebook

---

## Key Takeaways

- Built a complete ML pipeline from scratch
- Practiced proper train/test separation
- Learned model persistence and reuse
- Established a strong regression baseline
- Followed clean project structure and best practices

---

## Future Improvements

- Try non-linear models (Random Forest, Gradient Boosting)
- Hyperparameter tuning
- Feature engineering
- Convert model into an API (FastAPI / Flask)
- Add cross-validation

---

## Author

**Rizwan Likhon**  
Machine Learning & Data Science Projects
