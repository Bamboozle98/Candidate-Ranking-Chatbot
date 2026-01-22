# Bank Marketing Streamlit App  
## Interactive Predictive Modeling & What-If Analysis

This project implements an **end-to-end predictive modeling pipeline** for customer subscription behavior using the Bank Marketing dataset.  
It combines **model training**, **consistent preprocessing**, and an **interactive Streamlit dashboard** that allows users to explore the data and perform *what-if* predictions using a synthetic customer profile.

The target outcome is whether a customer **subscribes to a term deposit**, framed in an accessible, non-technical way for end users.

---

## 📘 Project Overview

The central goal of this project is to **predict customer subscription behavior** and provide an intuitive interface for understanding how individual features influence model predictions.

The repository is split into two tightly coupled components:

1. **Model Training Pipeline**
   - Feature engineering
   - Encoding and scaling
   - MLPClassifier training and export

2. **Streamlit Application**
   - Exploratory data analysis
   - Visualization and clustering
   - Interactive *Model Playground* for hypothetical customers

A key design principle is **training–inference consistency**:  
the Streamlit app loads the *exact* preprocessing pipeline and model artifacts saved during training.

---

## 🧾 Dataset Description

The dataset contains information related to customer marketing interactions, including:

- **Demographics** (e.g., age, job, marital status)
- **Financial attributes** (e.g., balance, loans)
- **Campaign details** (e.g., contact type, month)
- **Temporal indicators** (engineered features such as time bins)

### **Target Variable**

| Variable | Meaning |
|--------|--------|
| `y` | Whether the customer subscribed to a term deposit (`yes` / `no`) |

---

## ⚙️ Modeling Workflow

### 1. **Data Preparation**
- Raw data is loaded and cleaned using a shared data loader.
- Feature engineering steps (e.g., time binning) are applied consistently.
- Data is split into **training and test sets**, preserving class distribution where appropriate.

### 2. **Preprocessing**
A `ColumnTransformer` is used to ensure reproducibility:

- **Numerical features**
  - Standardized using `StandardScaler`
- **Categorical features**
  - One-hot encoded using `OneHotEncoder`

This preprocessing pipeline is **fit only on training data** and saved for inference.

---

## 🤖 Model Architecture

### **Multi-Layer Perceptron (MLPClassifier)**

- Feed-forward neural network for tabular classification
- Supports non-linear interactions between features
- Tuned using cross-validation during training

The trained model and fitted encoder are saved as artifacts and reused by the Streamlit app.

---

## 🔍 Model Playground (Streamlit)

One of the core features of this project is the **Model Playground**, an interactive interface that allows users to:

- Create an **imaginary customer**
- Adjust feature values using sliders and dropdowns
- Generate a prediction using the trained MLP
- Receive a **layman-friendly explanation** of the result

### Prediction Output

Instead of returning raw model labels (`y = 0` / `y = 1`), the app presents results as:

- A **clear True / False outcome**
- A natural-language explanation of whether the customer *would* or *would not* subscribe
- Optional probability estimates (when available)

This approach makes the model accessible to non-technical stakeholders while preserving correctness.

---

## 📊 Evaluation Metrics (Training)

During model development, multiple metrics are used to evaluate performance:

| Metric | Description |
|------|------------|
| **ROC AUC** | Ability to distinguish subscribers vs non-subscribers |
| **F1 Score** | Balance between precision and recall |
| **Precision** | Likelihood that a predicted subscriber truly subscribes |
| **Recall** | Ability to identify actual subscribers |

Metrics are computed on held-out test data to assess generalization.

---

## 🗂 Repository Structure

```pgsql
├── app.py                         # Streamlit application entrypoint
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── src/
│   ├── data/
│   │   └── LoadData.py             # Data loading and preprocessing
│   ├── models/
│   │   ├── neuralnetwork.py        # MLP model training script
│   │   ├── lightGPM.py             # Alternative model training script
│   │   └── saved/
│   │       ├── models/
│   │       │   └── mlp_model.joblib   # Saved MLP model
│   │       └── encoders/
│   │           └── encoder.joblib     # Saved preprocessing pipeline
│   └── features/
│       └── (optional helper modules)
└── notebooks/
    └── exploratory notebooks       # EDA and experimentation
```

## Streamlit
To run application locally, provided the environment has been created with the proper requirements, simply run

```bash
streamlit run app.py
```

To access the live hosted version of this application online visit [here](https://ppr9ytxpuuy9s8uh-mjda7wokmwjz8ehzsxvxms.streamlit.app).
