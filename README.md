# Cybersecurity Attack Classification Using Neural Networks

## Project Overview

This machine learning project focuses on **classifying cyberattack types** using a neural network built with Keras (JAX backend). The system analyzes a large-scale cybersecurity dataset containing **100,000 records** of cyberattack incidents across multiple industries, geographies, and attack vectors, and trains a multi-class classification model to predict the type of attack based on various network and contextual features.

## Dataset

The dataset (`dataset.csv`) contains 100,000 cyberattack records with the following **15 features**:

| Feature | Type | Description |
|---------|------|-------------|
| `attack_type` | Categorical | Type of cyberattack (target variable) — 8 classes including Phishing, DDoS, Zero-Day Exploit, SQL Injection, etc. |
| `target_system` | Categorical | System targeted (Cloud Service, Email Server, IoT Device, etc.) |
| `outcome` | Categorical | Attack result (Success / Failure) |
| `timestamp` | DateTime | When the attack occurred |
| `attacker_ip` | String | IP address of the attacker |
| `target_ip` | String | IP address of the target |
| `data_compromised_GB` | Numeric | Amount of data compromised (in GB) |
| `attack_duration_min` | Numeric | Duration of the attack (in minutes) |
| `security_tools_used` | Categorical | Defensive tools employed (Firewall, MFA, Endpoint Detection, etc.) |
| `user_role` | Categorical | Role of the affected user (Employee, Admin, Contractor, External User) |
| `location` | Categorical | Geographic location (Australia, Brazil, Germany, Russia, UK, etc.) |
| `attack_severity` | Numeric | Severity level (1–10) |
| `industry` | Categorical | Targeted industry (Energy, Retail, Finance, Healthcare, etc.) |
| `response_time_min` | Numeric | Incident response time (in minutes) |
| `mitigation_method` | Categorical | Mitigation strategy (Containment, Reset Credentials, Quarantine, Patch, etc.) |

## Project Pipeline

### 1. Exploratory Data Analysis (EDA)
**Notebook:** `ML_Project_Preprocessing.ipynb`

- Loads and inspects the raw dataset
- Displays dataset structure, shape, data types, and summary statistics
- Identifies missing values and data quality issues
- Visualizes distributions of numerical features (histograms for `data_compromised_GB`, `attack_duration_min`, `attack_severity`, `response_time_min`)
- Summarizes key insights about the data

### 2. Data Preprocessing
**Notebook:** `ML_Project_Preprocessing.ipynb`

- **Timestamp engineering:** Extracts temporal features — `year`, `month`, `day`, `day_of_week`, `hour`, `minute`, `is_weekend`, `is_night`, `is_business_hours`
- **Cyclical encoding:** Applies sine/cosine transformations to `hour`, `day_of_week`, and `month` for proper cyclical representation
- **IP address parsing:** Decomposes `attacker_ip` and `target_ip` into individual octets (`oct_1` through `oct_4`)
- **Categorical encoding:** Label-encodes categorical features (`attack_type`, `target_system`, `security_tools_used`, `user_role`, `mitigation_method`)
- **Target variable creation:** Creates a binary `target` column from the `outcome` field (Success/Failure)
- **Feature selection:** Drops raw columns that have been engineered (`timestamp`, `attacker_ip`, `target_ip`, `outcome`, `location`, `industry`)
- Exports the cleaned data to `preprocessed_data_final.csv` (33 features, all numeric)

### 3. Model Implementation
**Notebook:** `Model_Implement.ipynb`

- **Framework:** Keras 3.x with JAX backend
- **Task:** Multi-class classification (8 attack types)
- **Architecture:** Sequential neural network
  - Dense(32, ReLU) → Dropout(0.3)
  - Dense(16, ReLU) → Dropout(0.2)
  - Dense(8, Softmax)
  - **Total parameters:** 1,688
- **Feature scaling:** StandardScaler normalization
- **Train/Test split:** 80/20 with stratification
- **Loss function:** Sparse Categorical Cross-entropy
- **Optimizer:** Adam (learning_rate=0.001)
- **Metrics:** Accuracy

### 4. Base Model / Comparison
**Notebook:** `22I-162_22I1722_BaseModel.ipynb`

- Performs a parallel EDA and preprocessing pipeline
- Serves as a baseline comparison for the improved model implementation

## Key Findings

- The dataset is **well-balanced** across all 8 attack types (~12,500 samples each)
- Numerical features (`data_compromised_GB`, `attack_duration_min`, `attack_severity`, `response_time_min`) show **approximately uniform distributions**
- Feature-target correlations are **very weak** (max ~0.01), indicating that individual features have limited predictive power — the model must learn complex non-linear feature interactions
- Baseline (most-frequent class) accuracy is approximately **12.6%**, which the neural network needs to significantly exceed

## Project Structure

```
ML Project/
├── dataset.csv                          # Raw dataset (100K records, 15 columns)
├── ML_Project_Preprocessing.ipynb       # EDA and data preprocessing notebook
├── Model_Implement.ipynb                # Neural network model training and evaluation
├── 22I-162_22I1722_BaseModel.ipynb      # Base model for comparison
├── preprocessed_cybersecurity_data.csv  # Intermediate preprocessed data
├── preprocessed_data_final.csv          # Final preprocessed data (33 numeric features)
├── logistic_regression_nn_model.keras   # Saved trained model
├── model_metrics.pkl                    # Saved model performance metrics
├── model_predictions.csv               # Model predictions on test set
├── training_history.pkl                 # Training history (loss/accuracy curves)
├── README.md                           # Project description (this file)
└── .venv/ / venv/                      # Python virtual environments
```

## Technologies Used

- **Python 3.x**
- **Keras 3.x** (with JAX backend) — Neural network framework
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computing
- **Scikit-learn** — Preprocessing (StandardScaler, train_test_split), metrics (accuracy, classification report, confusion matrix, ROC-AUC)
- **Matplotlib / Seaborn** — Data visualization
- **Google Colab** — Development environment

## How to Run

1. **Set up the environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   pip install pandas numpy keras jax scikit-learn matplotlib seaborn
   ```

2. **Run the preprocessing notebook:**
   Open `ML_Project_Preprocessing.ipynb` and execute all cells to generate `preprocessed_data_final.csv`.

3. **Run the model training notebook:**
   Open `Model_Implement.ipynb` and execute all cells to train the neural network and evaluate performance.

## Authors

- **22I-162**
- **22I-1722**
