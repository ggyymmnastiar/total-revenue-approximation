# 💲 Total Revenue Approximation

A Machine Learning web application that predicts the **approximate total revenue** (in IDR) from dairy products using a **Random Forest Regressor**, deployed through an interactive Streamlit dashboard.

🔗 **Live Dashboard:** [g-revenue.streamlit.app](https://g-revenue.streamlit.app/)

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset Description](#-dataset-description)
3. [Model & Machine Learning Pipeline](#-model--machine-learning-pipeline)
4. [Dashboard Capabilities](#-dashboard-capabilities)
5. [Getting Started](#-getting-started)
6. [Repository Structure](#-repository-structure)
7. [Tech Stack](#-tech-stack)

---

## 📌 Project Overview

Revenue forecasting is a key operational challenge for dairy farms — accurate estimates help inform pricing strategies, production planning, and stock management decisions.

This project builds an end-to-end regression system that approximates total dairy product revenue based on operational and sales attributes. Users can interact with the dashboard to explore the dataset, train the model on demand, and obtain an instant revenue estimate by configuring input parameters through the sidebar.

> Currency conversion is applied from Indian Rupee (INR) to Indonesian Rupiah (IDR) at a fixed exchange rate of **1 INR = 185 IDR**, making the output directly relevant for the Indonesian market context.

### 🔄 End-to-End Workflow

```
Raw Dataset (4,325 samples from Indian dairy farms)
    ↓
Currency Conversion (INR → IDR)
    ↓
Feature Selection & Missing Value Check
    ↓
Preprocessing Pipeline (StandardScaler + OneHotEncoder)
    ↓
Random Forest Regressor + Hyperparameter Tuning
    ↓
Interactive Streamlit Dashboard (Real-time revenue prediction)
```

---

## 📊 Dataset Description

The dataset contains **4,325 records** of dairy farm transactions sourced from Indian dairy operations. The original target variable `Approx. Total Revenue(INR)` is converted to IDR for the prediction output.

### All Available Columns

| Column | Type | Description |
|--------|------|-------------|
| `Location` | Categorical | Indian state where the farm is located |
| `Total Land Area (acres)` | Numerical | Total farm land area |
| `Number of Cows` | Numerical | Number of cows on the farm |
| `Farm Size` | Categorical | Farm size category (Small / Medium / Large) |
| `Date` | Date | Transaction / recording date |
| `Product ID` | Numerical | Unique product identifier |
| `Product Name` | Categorical | Type of dairy product (Milk, Cheese, Yogurt, etc.) |
| `Brand` | Categorical | Dairy brand name |
| `Quantity (liters/kg)` | Numerical | Total quantity produced |
| `Price per Unit` | Numerical | Purchase price per unit (INR) |
| `Total Value` | Numerical | Total value of produced quantity (INR) |
| `Shelf Life (days)` | Numerical | Product shelf life in days |
| `Storage Condition` | Categorical | Storage method (Refrigerated, Frozen, Ambient, etc.) |
| `Production Date` | Date | Product production date |
| `Expiration Date` | Date | Product expiration date |
| `Quantity Sold (liters/kg)` | Numerical | Quantity actually sold |
| `Price per Unit (sold)` | Numerical | Selling price per unit (INR) |
| `Approx. Total Revenue(INR)` | Numerical | **Target variable** — approximate revenue (INR) |
| `Customer Location` | Categorical | Buyer's state location |
| `Sales Channel` | Categorical | Channel of sale (Retail / Wholesale / Online) |
| `Quantity in Stock (liters/kg)` | Numerical | Remaining inventory |
| `Minimum Stock Threshold (liters/kg)` | Numerical | Reorder trigger threshold |
| `Reorder Quantity (liters/kg)` | Numerical | Quantity to reorder |

### Features Used for Modeling

The model uses the following **9 input features** (after feature selection):

| Feature | Type | Description |
|---------|------|-------------|
| `Number of Cows` | Numerical | Number of cows on the farm |
| `Farm Size` | Categorical | Small / Medium / Large |
| `Product Name` | Categorical | Milk / Cheese / Yoghurt |
| `Quantity (liters/kg)` | Numerical | Total quantity produced |
| `Shelf Life (days)` | Numerical | Product shelf life |
| `Storage Condition` | Categorical | Refrigerated / Frozen / Ambient / Polythene Packet / Tetra Pack |
| `Quantity Sold (liters/kg)` | Numerical | Quantity sold |
| `Sales Channel` | Categorical | Retail / Wholesale / Online |
| `Price per Unit (sold) (IDR)` | Numerical | Selling price per unit (converted to IDR) |

**Target:** `Approx. Total Revenue (IDR)` — converted from INR at 1 INR = 185 IDR

---

## 🤖 Model & Machine Learning Pipeline

**Algorithm:** Random Forest Regressor — an ensemble of decision trees that averages multiple predictions to produce a robust continuous output, reducing overfitting compared to a single decision tree.

### Preprocessing Pipeline

Built with `sklearn.compose.ColumnTransformer` and `sklearn.pipeline.Pipeline`:

```
Input Features
    ↓
ColumnTransformer
    ├── Numerical (5 features) → StandardScaler()
    └── Categorical (4 features) → OneHotEncoder(handle_unknown='ignore')
    ↓
RandomForestRegressor()
    ↓
Predicted Approx. Total Revenue (IDR)
```

### Model Evaluation Metrics

After training via the dashboard's interactive **"Train the Data (.fit)"** button, the following metrics are displayed:

| Metric | Description |
|--------|-------------|
| **R² (R-squared)** | Proportion of variance explained by the model (closer to 1.0 is better) |
| **MAPE** | Mean Absolute Percentage Error — average percentage deviation from actual values |
| **MAE** | Mean Absolute Error in Rupiah — average absolute prediction error |
| **RMSE** | Root Mean Squared Error — penalizes larger errors more heavily |

A scatter plot of **Actual vs. Predicted Revenue** is also rendered post-training to visually assess model fit.

### Train / Test Split

- **Train:** 80% of data (`random_state=101`)
- **Test:** 20% of data

---

## 🖥️ Dashboard Capabilities

| Section | Content & Functionality |
|---------|-------------------------|
| **Data Overview** | Full raw dataset preview, input feature matrix (X) |
| **Data Preparation & EDA** | Currency conversion log, missing value check, feature selection definition |
| **Data Visualization** | *(Coming soon)* |
| **Input Features** | Current sidebar input, combined input with dataset preview |
| **Data Preprocessing** | Pipeline architecture code, train/test split stats, interactive training button with evaluation metrics (R², MAPE, MAE, RMSE) and Actual vs. Predicted scatter plot |
| **Hyperparameter Tuning** | *(Coming soon)* |
| **Prediction Result** | Real-time revenue approximation displayed in IDR based on sidebar inputs |

Users adjust input values through the **sidebar** using sliders, dropdowns, and number inputs, and the predicted revenue updates instantly.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ggyymmnastiar/total-revenue-approximation.git
   cd total-revenue-approximation
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS / Linux
   # .venv\Scripts\activate       # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run streamlit_app.py
   ```

5. Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Repository Structure

```
total-revenue-approximation/
├── .devcontainer/
│   └── devcontainer.json       # GitHub Codespaces configuration
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
├── dairy_dataset.csv           # Dairy farm transactions dataset (4,325 records)
├── streamlit_app.py            # Main Streamlit application
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🛠️ Tech Stack

- **Application Framework:** Streamlit
- **Machine Learning:** scikit-learn (RandomForestRegressor, Pipeline, ColumnTransformer)
- **Data Processing:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Language:** Python 3

---

## 📄 License

MIT
