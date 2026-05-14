# Customer Retention Analytics - Setup & Execution Guide

## 🎯 Quick Start (10 minutes)

### Step 1: Create Project Directory Structure

```bash
# Create main project folder
mkdir customer-retention-analytics
cd customer-retention-analytics

# Create subdirectories
mkdir -p data/raw data/processed
mkdir -p models results configs app
mkdir -p notebooks tests src logs
```

### Step 2: Copy All Files

Place the provided Python files in the correct locations:

```
customer-retention-analytics/
├── requirements.txt          (install dependencies)
├── data_generation.py         → src/data_generation.py
├── feature_engineering.py     → src/feature_engineering.py
├── causal_estimation.py       → src/causal_estimation.py
├── validation.py              → src/validation.py
├── dashboard.py               → app/dashboard.py
└── README.md                  (documentation)
```

### Step 3: Create Virtual Environment

```bash
# Python 3.9+
python -m venv venv

# Activate
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import econml; import xgboost; print('✅ Installation successful')"
```

---

## 🚀 Full Pipeline Execution (Sequential)

### Phase 1: Data Generation (2-3 minutes)

Generate synthetic retail dataset with 100K customers:

```bash
python src/data_generation.py \
  --output data/raw/synthetic_retail.csv \
  --n_samples 100000 \
  --seed 42
```

**Expected Output:**
```
Generating 100000 customer records...
Data saved to data/raw/synthetic_retail.csv
File size: XX.XX MB

DATASET SUMMARY
============================================================
  customer_id  age  tenure_months  annual_income  ...
0           1   42             24          75423  ...
1           2   58             18          94521  ...
...
```

**Check:** Open `data/raw/synthetic_retail.csv` to verify (should be ~100K rows, 30+ columns)

---

### Phase 2: Feature Engineering (2-3 minutes)

Preprocess and create 80+ features:

```bash
python src/feature_engineering.py \
  --input data/raw/synthetic_retail.csv \
  --output data/processed/features_engineered.csv \
  --preprocessor models/preprocessor.pkl
```

**Expected Output:**
```
Loading data from data/raw/synthetic_retail.csv
Loaded 100000 rows, 30 columns
Filled [missing columns]
Created XX interaction features
Created XX aggregate features
Features engineered: 85 features
Output saved to data/processed/features_engineered.csv
```

**Check:** `data/processed/features_engineered.csv` should have ~85 feature columns + targets

---

### Phase 3: Causal Model Training (5-10 minutes)

Train T-Learner, X-Learner, Causal Forest, and ensemble:

```bash
python src/causal_estimation.py \
  --data data/processed/features_engineered.csv \
  --output models/
```

**Expected Output:**
```
Loading data from data/processed/features_engineered.csv
Loaded 100000 rows, 90 columns

============================================================
Training T-Learner
============================================================
T-Learner CATE shape: (100000,)
T-Learner CATE mean: -0.1234
T-Learner CATE std: 0.0567

============================================================
Training X-Learner
============================================================
X-Learner CATE shape: (100000,)
X-Learner CATE mean: -0.1189
X-Learner CATE std: 0.0543

============================================================
Training Causal Forest
============================================================
Causal Forest CATE shape: (100000,)
Causal Forest CATE mean: -0.1256
Causal Forest CATE std: 0.0612

============================================================
CAUSAL ESTIMATION SUMMARY
============================================================

T_LEARNER
  Mean effect: -0.1234
  Std effect: 0.0567
  Range: [-0.3456, 0.0123]
  Percentiles (p10, p50, p90): -0.2145, -0.1089, -0.0234

X_LEARNER
  Mean effect: -0.1189
  Std effect: 0.0543
  Range: [-0.3234, 0.0456]
  Percentiles (p10, p50, p90): -0.2012, -0.1034, -0.0156

CAUSAL_FOREST
  Mean effect: -0.1256
  Std effect: 0.0612
  Range: [-0.3890, 0.0789]
  Percentiles (p10, p50, p90): -0.2289, -0.1145, -0.0312

ENSEMBLE
  Mean effect: -0.1226
  Std effect: 0.0574
  Range: [-0.3527, 0.0456]
  Percentiles (p10, p50, p90): -0.2149, -0.1089, -0.0234

Summary saved to models/causal_summary.csv
```

**Check:** 
- `models/t_learner.pkl`, `models/x_learner.pkl`, etc. should exist
- `models/causal_summary.csv` should contain model metrics

---

### Phase 4: Validation (3-5 minutes)

Validate CATE estimates with Qini curves, AUUC, and placebo tests:

```bash
python src/validation.py \
  --data data/processed/features_engineered.csv \
  --output results/
```

**Expected Output:**
```
Loading data from data/processed/features_engineered.csv
Loaded 100000 rows, 90 columns

Qini Curve Analysis:
Qini AUUC: 0.0456

CINI Curve Analysis:
[CINI statistics]

Placebo Test:
Running placebo test...
Real AUUC: 0.0456
Placebo AUUC (mean): 0.0021
Placebo AUUC (std): 0.0045
P-value: 0.0001
Significant: True

Segment Heterogeneity Analysis:
  vip: mean=-0.1123, std=0.0445
  loyal: mean=-0.1289, std=0.0612
  at_risk: mean=-0.1456, std=0.0734
  dormant: mean=-0.0856, std=0.0312

Identified 25000 responders (top 25%)
Mean CATE for responders: -0.1834
Mean CATE for non-responders: -0.0618

Results saved to results/
```

**Check:**
- `results/validation_results.json` contains all metrics
- `results/high_responders.csv` identifies top-25% treatment responders

---

### Phase 5: Launch Interactive Dashboard

Start the Streamlit app:

```bash
streamlit run app/dashboard.py
```

**Expected Output:**
```
Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.XX:8501

  For better performance, install the watchdog module:

  $ pip install watchdog
```

**Then:**
1. Open http://localhost:8501 in your browser
2. Use sidebar to navigate between pages
3. Explore interactive visualizations and policy simulator

---

## 📊 Dashboard Pages Explained

### 1. **📈 Overview**
- High-level KPIs (customers, churn rate, mean CATE)
- Model architecture diagram
- Key findings summary

### 2. **🔍 CATE Analysis**
- Distribution of treatment effects
- Heterogeneity by customer segment
- Statistical summaries per segment

### 3. **📊 Model Comparison**
- Performance metrics for all causal models
- Model descriptions and approaches
- Detailed statistics table

### 4. **🎯 Policy Simulator**
- Interactive targeting policy curve
- Adjust targeting % and cost per treatment
- Real-time ROI calculations
- Expected retention lift

### 5. **✅ Validation**
- Qini curve with AUUC metric
- Placebo test results (p-value)
- Statistical validation summary

---

## 🔧 Configuration & Customization

### Modify Data Generation Parameters

Edit the call to `data_generation.py`:

```bash
python src/data_generation.py \
  --n_samples 50000 \
  --seed 123
```

### Adjust Causal Model Parameters

In `causal_estimation.py`, modify XGBoost hyperparameters:

```python
estimator = TLearner(
    models=xgb.XGBRegressor(
        max_depth=5,           # Increase for more complexity
        n_estimators=100,      # More trees = slower but potentially better
        learning_rate=0.1,     # Lower = slower training, better generalization
        ...
    )
)
```

### Change Validation Parameters

In `validation.py`:

```python
validator.placebo_test(y, T, true_cate, n_iterations=200)  # More iterations for robustness
```

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'econml'"

**Solution:**
```bash
pip install econml --upgrade
# If still fails, install dependencies separately
pip install scikit-learn numpy scipy pandas
```

### Problem: "File not found: data/raw/synthetic_retail.csv"

**Solution:**
```bash
# Run data generation first
python src/data_generation.py --output data/raw/synthetic_retail.csv
```

### Problem: Streamlit app won't start

**Solution:**
```bash
# Check if port 8501 is in use
streamlit run app/dashboard.py --server.port 8502

# Or clear cache
streamlit cache clear
streamlit run app/dashboard.py
```

### Problem: Out of memory with large datasets

**Solution:**
```bash
# Reduce sample size
python src/data_generation.py --n_samples 50000

# Or increase available memory
# In Windows: Control Panel > System > Advanced > Environment Variables
# In macOS/Linux: ulimit -v
```

### Problem: XGBoost warnings about GPU

**Solution:**
```bash
# Disable GPU training (use CPU)
# Add to causal_estimation.py:
# tree_method='hist'  # CPU-only
```

---

## 📈 Pipeline Performance Benchmarks

| Phase | Typical Time | CPU | Memory |
|-------|-------------|-----|--------|
| Data Generation | 2-3 min | 1 core | 2 GB |
| Feature Engineering | 2-3 min | 2 cores | 3 GB |
| Causal Training | 5-10 min | 4 cores | 4 GB |
| Validation | 3-5 min | 4 cores | 2 GB |
| Dashboard Startup | <5 sec | 1 core | 1 GB |
| **Total** | **15-30 min** | **4 cores** | **4 GB** |

---

## 🎯 Expected Output Files

After running all phases:

```
customer-retention-analytics/
├── data/
│   ├── raw/
│   │   └── synthetic_retail.csv         (100K rows, 30 cols) ~30 MB
│   └── processed/
│       └── features_engineered.csv      (100K rows, 85 cols) ~45 MB
├── models/
│   ├── preprocessor.pkl                 (scaler, encoders)
│   ├── t_learner.pkl                    (trained model)
│   ├── x_learner.pkl                    (trained model)
│   ├── causal_forest.pkl                (trained model)
│   └── causal_summary.csv               (model metrics)
├── results/
│   ├── validation_results.json          (Qini, AUUC, placebo)
│   └── high_responders.csv              (top 25% customers)
└── logs/
    └── execution.log                    (detailed logs)
```

---

## 🚢 Production Deployment

### REST API (Optional)

Create `app/api.py`:

```python
from fastapi import FastAPI
import pickle

app = FastAPI()

# Load model
with open('models/causal_forest.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(features: dict):
    # Preprocess and predict CATE
    cate = model.predict(features)
    return {"cate": float(cate)}
```

Run with:
```bash
uvicorn app/api:app --host 0.0.0.0 --port 8000
```

### Docker (Optional)

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app/dashboard.py"]
```

Build and run:
```bash
docker build -t retention-analytics .
docker run -p 8501:8501 retention-analytics
```

---

## 📚 Further Reading

- [EconML Documentation](https://econml.azurewebsites.net/)
- [Heterogeneous Treatment Effects](https://arxiv.org/abs/1701.08835)
- [Qini Curve & AUUC](https://projecteuclid.org/euclid.aoas/1316750699)
- [DoWhy Causal Inference](https://microsoft.github.io/dowhy/)

---

## 🎓 Learning Outcomes

After completing this project, you'll understand:

✅ Causal inference methodology (T-Learner, X-Learner)
✅ Heterogeneous treatment effect estimation
✅ CATE validation (Qini, AUUC, placebo tests)
✅ Personalized marketing optimization
✅ Interactive dashboard development
✅ Production ML pipelines

---

## 📞 Support

If you encounter issues:

1. Check the **Troubleshooting** section above
2. Verify file paths match your directory structure
3. Ensure Python version is 3.9+
4. Run one step at a time and check output
5. Check log files in `logs/` directory

---

**Good luck with your Customer Retention Analytics project! 🚀**

Last Updated: May 2026 | Version 1.0.0
