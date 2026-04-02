# INTRUSION DETECTION SYSTEM (IDS) - ML PROJECT IMPLEMENTATION PLAN
## UNSW-NB15 Dataset | Multi-Class Classification (9 Attack Types + Normal)

---

## PROJECT OVERVIEW

**Objective**: Build a complete ML pipeline for network intrusion detection with multi-class classification  
**Dataset**: UNSW-NB15 (CSV format)  
**Target**: 9 attack types + Normal traffic classification  
**Platform**: Google Colab with Google Drive integration  
**Deliverables**: 4 ML models, SHAP explainability, Streamlit UI (3 pages)

---

## PHASE 1: ENVIRONMENT SETUP & DATA INGESTION
**Duration**: Day 1, Step 1  
**Tag**: `Setup`

### Tasks:
1. **Mount Google Drive in Colab**
   - Install required packages: `pip install pyarrow fastparquet snappy`
   - Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`
   - Create project directory structure:
     ```
     /content/drive/MyDrive/ids-project/
     ├── data/
     ├── models/
     ├── artifacts/
     └── outputs/
     ```

2. **Set Global Random State**
   - Set `random_state=42` globally for reproducibility
   - Apply to: numpy, random, sklearn, model training

3. **Install Additional Libraries**
   ```python
   pip install lightgbm xgboost imbalanced-learn shap streamlit
   ```

### Deliverables:
- ✅ Ready environment confirmation message
- ✅ Directory structure created
- ✅ All libraries installed and imported

---

## PHASE 2: CSV TO PARQUET CONVERSION
**Duration**: Day 1, Step 2  
**Tag**: `Storage`

### Tasks:
1. **Read UNSW-NB15 CSV Files**
   - Load CSV in 500k-row chunks to manage memory
   - Strip whitespace from column names: `df.columns = df.columns.str.strip()`

2. **Save as Parquet with Snappy Compression**
   ```python
   import pyarrow as pa
   import pyarrow.parquet as pq
   
   # Save with snappy compression
   df.to_parquet('unsw_nb15.parquet', 
                  engine='pyarrow', 
                  compression='snappy',
                  index=False)
   ```

3. **Verify Conversion**
   - Compare file sizes (CSV vs Parquet)
   - Verify data integrity: row count, column names, data types
   - Test read speed comparison

### Deliverables:
- ✅ `unsw_nb15.parquet` file (~3x faster for all future reads)
- ✅ Conversion report with size/speed metrics

---

## PHASE 3: DATA CLEANING
**Duration**: Day 1, Step 3  
**Tag**: `Quality`

### Tasks:
1. **Handle Missing Values**
   - **Strategy**: 
     - Replace `inf` → `NaN`
     - Drop columns with >50% missing values
     - Median imputation for numerical features
     - Remove exact duplicate rows
   
   ```python
   from sklearn.impute import SimpleImputer
   
   # Median imputation
   imputer = SimpleImputer(strategy='median')
   df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
   ```

2. **Fix Data Type Issues**
   - Ensure numeric columns are `float64` or `int64`
   - Fix any mixed-type columns

3. **Standardize Labels**
   - Clean label strings (lowercase, strip spaces)
   - Create clean label mapping

4. **Remove Duplicates**
   - Drop exact duplicate rows
   - Log the count of removed duplicates

### Deliverables:
- ✅ Clean typed DataFrame
- ✅ Before/after row counts
- ✅ Missing value report

---

## PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)
**Duration**: Day 1, Step 4  
**Tag**: `EDA`

### Tasks:
1. **Class Distribution Analysis**
   - **Bar Chart**: Show count of each attack type + Normal
   - Save as: `class_distribution.png`
   - Note class imbalance for later handling

2. **Correlation Analysis**
   - **Pearson Correlation Heatmap** (top 40 features)
   - Identify highly correlated features (|r| > 0.9)
   - Save as: `correlation_heatmap.png`

3. **Missing Value Visualization**
   - **Heatmap**: Show missing value patterns
   - Save as: `missing_values_map.png`

4. **Feature Distribution Analysis**
   - **Box plots**: Top 10 features by variance
   - Identify outliers
   - Save as: `top10_feature_boxplots.png`

5. **Save All Figures to Google Drive**
   ```python
   import matplotlib.pyplot as plt
   plt.savefig('/content/drive/MyDrive/ids-project/artifacts/plot_name.png', 
               dpi=300, bbox_inches='tight')
   ```

### Key Insights to Document:
- Class imbalance ratio
- Number of highly correlated feature pairs
- Outlier percentage per feature

### Deliverables:
- ✅ 4 EDA visualizations saved to Drive
- ✅ 5-6 EDA charts confirmed in Colab

---

## PHASE 5: FEATURE ENGINEERING & SELECTION
**Duration**: Day 1, Step 5  
**Tag**: `Features`

### Tasks:
1. **Handle Multicollinearity**
   - **Remove one feature from each correlated pair** (|ρ| > 0.95)
   - Use VIF (Variance Inflation Factor) if needed
   - Document removed features

2. **Feature Selection: SelectKBest**
   ```python
   from sklearn.feature_selection import SelectKBest, mutual_info_classif
   
   selector = SelectKBest(score_func=mutual_info_classif, k=30)
   X_selected = selector.fit_transform(X, y)
   
   # Get selected feature names
   selected_features = X.columns[selector.get_support()].tolist()
   ```

3. **Reduce to 30 Features**
   - Save final feature list as: `feature_names.pkl`
   - Document feature importance scores

4. **Create Feature Metadata**
   ```python
   import pickle
   
   feature_info = {
       'selected_features': selected_features,
       'removed_correlated': removed_features,
       'total_original': len(X.columns)
   }
   
   with open('feature_names.pkl', 'wb') as f:
       pickle.dump(feature_info, f)
   ```

### Deliverables:
- ✅ `feature_names.pkl` (reduced from ~43 cols to 30)
- ✅ Feature selection report

---

## PHASE 6: PREPROCESSING & DATA SPLIT
**Duration**: Day 1, Step 6  
**Tag**: `Prep`

### Tasks:
1. **Label Encoding for Multi-Class Target**
   ```python
   from sklearn.preprocessing import LabelEncoder
   
   label_encoder = LabelEncoder()
   y_encoded = label_encoder.fit_transform(y)
   
   # Save encoder
   with open('label_encoder.pkl', 'wb') as f:
       pickle.dump(label_encoder, f)
   ```

2. **Stratified Train-Test Split (80/20)**
   ```python
   from sklearn.model_selection import train_test_split
   
   X_train, X_test, y_train, y_test = train_test_split(
       X_selected, y_encoded, 
       test_size=0.20, 
       stratify=y_encoded,
       random_state=42
   )
   ```

3. **Handle Class Imbalance**
   - **Use Stratified K-Fold CV** during training
   - **Apply class_weight='balanced'** for tree models
   - Consider SMOTE if severe imbalance exists

4. **Feature Scaling (StandardScaler)**
   ```python
   from sklearn.preprocessing import StandardScaler
   
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   
   # Save scaler
   with open('scaler.pkl', 'wb') as f:
       pickle.dump(scaler, f)
   ```

### Deliverables:
- ✅ `X_train_test_scaler.pkl` (preprocessed splits)
- ✅ `label_encoder.pkl` and `scaler.pkl`

---

## PHASE 7: MODEL TRAINING (4 MODELS)
**Duration**: Day 2, Step 7  
**Tag**: `Training`

### Models to Train:

#### **Model 1: Logistic Regression (Baseline)**
```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    solver='lbfgs',
    class_weight='balanced',
    random_state=42
)
lr_model.fit(X_train_scaled, y_train)
```

#### **Model 2: Random Forest (200 trees)**
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
```

#### **Model 3: XGBoost (300 rounds)**
```python
import xgboost as xgb

# Calculate scale_pos_weight for each class
class_weights = compute_sample_weight('balanced', y_train)

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=10,  # 9 attacks + 1 normal
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_scaled, y_train)
```

#### **Model 4: LightGBM (500 leaves)**
```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    num_leaves=500,
    max_depth=-1,
    learning_rate=0.05,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train_scaled, y_train)
```

### Training Strategy:
- **Use 5-fold Stratified Cross-Validation** for each model
- Record training time for each model
- Save trained models as `.pkl` files

### Deliverables:
- ✅ 4 model objects saved: `model_lgbm.pkl`, `scaler.pkl`, `label_encoder.pkl`
- ✅ CV scores for each model
- ✅ Training times logged

---

## PHASE 8: MODEL EVALUATION & COMPARISON
**Duration**: Day 2, Step 8  
**Tag**: `Results`

### Metrics to Calculate (Per Model):

1. **Accuracy**: Overall correct predictions
2. **Macro F1-Score**: Average F1 across all classes
3. **Per-Class F1-Score**: F1 for each attack type
4. **ROC-AUC (OvR)**: One-vs-Rest AUC for multi-class
5. **FAR (False Alarm Rate)**: FP / (FP + TN) per class
6. **Confusion Matrix**: 10x10 heatmap

```python
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    roc_auc_score, confusion_matrix
)

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
per_class_f1 = f1_score(y_test, y_pred, average=None)

# ROC-AUC (One-vs-Rest)
roc_auc = roc_auc_score(y_test, y_pred_proba, 
                         multi_class='ovr', average='macro')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
```

### Visualizations to Create:

1. **Confusion Matrix Heatmap** (per model)
   - 10x10 matrix with attack labels
   - Save as: `confusion_matrix_{model_name}.png`

2. **ROC Curve Overlay** (all 4 models)
   - Micro-average ROC curves
   - Save as: `roc_curves.png`

3. **Model Comparison Bar Chart**
   - Metrics: Accuracy, Macro F1, ROC-AUC
   - Save as: `model_comparison.png`

4. **Per-Class F1 Heatmap**
   - Rows: Models, Columns: Attack classes
   - Save as: `per_class_f1_heatmap.png`

### Save All Metrics
```python
import json

metrics_dict = {
    'LogisticRegression': {
        'accuracy': 0.XX,
        'macro_f1': 0.XX,
        'roc_auc': 0.XX,
        'per_class_f1': [...],
        'training_time': 'XX sec'
    },
    # ... for all 4 models
}

with open('metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=2)
```

### Deliverables:
- ✅ `metrics.json` with all model metrics
- ✅ `confusion_matrix.png`, `roc_curves.png`
- ✅ 6 saved plots total

---

## PHASE 9: SHAP EXPLAINABILITY ANALYSIS
**Duration**: Day 2, Step 9  
**Tag**: `Export`

### Tasks:

1. **Run SHAP TreeExplainer on LightGBM** (best model)
   ```python
   import shap
   
   # Use 450 held-out samples (50 per class)
   explainer = shap.TreeExplainer(lgb_model)
   shap_values = explainer.shap_values(X_test_scaled[:450])
   ```

2. **Generate SHAP Visualizations**

   a. **SHAP Summary Plot** (Feature Importance)
   ```python
   shap.summary_plot(shap_values, X_test_scaled[:450], 
                     feature_names=selected_features,
                     show=False)
   plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
   ```

   b. **SHAP Waterfall Chart** (Single Prediction)
   ```python
   # Pick one sample from each attack type
   shap.plots.waterfall(shap_values[sample_idx], show=False)
   plt.savefig('shap_waterfall_chart.png', dpi=300, bbox_inches='tight')
   ```

3. **Save SHAP Values for Streamlit**
   ```python
   shap_data = {
       'shap_values': shap_values,
       'test_samples': X_test_scaled[:450],
       'feature_names': selected_features
   }
   
   with open('shap_values.pkl', 'wb') as f:
       pickle.dump(shap_data, f)
   ```

4. **Export Test Samples CSV**
   ```python
   test_samples_df = pd.DataFrame(
       X_test_scaled[:450], 
       columns=selected_features
   )
   test_samples_df['attack_type'] = label_encoder.inverse_transform(y_test[:450])
   test_samples_df.to_csv('test_samples.csv', index=False)
   ```

5. **Write Attack Descriptions JSON**
   ```python
   attack_info = {
       "Normal": "Legitimate network traffic",
       "Fuzzers": "Attempts to cause crashes by sending random data",
       "Analysis": "Port scanning and reconnaissance attacks",
       "Backdoors": "Unauthorized access attempts",
       "DoS": "Denial of Service attacks",
       "Exploits": "Exploitation of known vulnerabilities",
       "Generic": "Generic attacks against block ciphers",
       "Reconnaissance": "Information gathering attacks",
       "Shellcode": "Exploits that execute arbitrary code",
       "Worms": "Self-replicating malware"
   }
   
   with open('attack_info.json', 'w') as f:
       json.dump(attack_info, f, indent=2)
   ```

### Deliverables:
- ✅ `shap_values.pkl` for top 450 samples
- ✅ `test_samples.csv` with attack types
- ✅ `attack_info.json` with descriptions
- ✅ All 8 artifacts ready in `/data/` (Colab.done message)

---

## PHASE 10: STREAMLIT UI DEVELOPMENT
**Duration**: Day 2, Steps 10-11  
**Tag**: `Streamlit UI`

### Architecture: 3-Page Streamlit App

#### **File Structure**
```
ids-project/
├── app/
│   ├── app.py                    # Main Streamlit entry point
│   ├── pages/
│   │   ├── 1_dashboard.py        # Page 1: Dashboard
│   │   ├── 2_live_simulation.py  # Page 2: Live Simulation
│   │   └── 3_explain_prediction.py # Page 3: SHAP Explainer
│   └── data/                     # All artifacts from Colab
│       ├── model_lgbm.pkl
│       ├── scaler.pkl
│       ├── label_encoder.pkl
│       ├── test_samples.csv
│       ├── shap_values.pkl
│       └── attack_info.json
└── requirements.txt
```

---

### **PAGE 1: DASHBOARD** 
**File**: `pages/1_dashboard.py`

#### Features:
1. **4 Metric Cards** (Top Row)
   - F1 Score (Macro)
   - FAR (False Alarm Rate)
   - Recall (Macro)
   - Accuracy

2. **Model Comparison Bar Chart** (All 4 Models)
   - X-axis: Models (LR, RF, XGB, LightGBM)
   - Y-axis: Metrics (Accuracy, F1, Recall)
   - Interactive Plotly chart

3. **Attack Class Distribution Donut Chart**
   - Show percentage of each attack type in test set
   - Use Plotly pie chart with hole=0.4

4. **Per-Class F1 Heatmap Table**
   - Rows: 4 models
   - Columns: 10 attack classes
   - Color-coded by F1 score

5. **Confusion Matrix** (Best Model - LightGBM)
   - 10x10 heatmap
   - Plotly heatmap with annotations

6. **ROC Curve Overlay** (All 4 Models)
   - Micro-average ROC
   - Show AUC in legend

#### Code Structure:
```python
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="IDS Dashboard", layout="wide")

# Load metrics
with open('data/metrics.json', 'r') as f:
    metrics = json.load(f)

# Metric cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("F1 Score", f"{metrics['LightGBM']['macro_f1']:.3f}")
col2.metric("FAR", f"{metrics['LightGBM']['far']:.3f}")
col3.metric("Recall", f"{metrics['LightGBM']['recall']:.3f}")
col4.metric("Accuracy", f"{metrics['LightGBM']['accuracy']:.3f}")

# Model comparison bar chart
fig = px.bar(comparison_df, x='Model', y='Metric', color='Type')
st.plotly_chart(fig, use_container_width=True)

# ... (continue for all 6 visualizations)
```

---

### **PAGE 2: LIVE SIMULATION**
**File**: `pages/2_live_simulation.py`

#### Features:
1. **Attack Type Selector** (4 Buttons)
   - Dropdown or button group to select from 10 attack types
   - On selection, filter `test_samples.csv` to get samples of that type

2. **Animated Packet Stream** (Real-time Effect)
   ```python
   import time
   
   # Create placeholder for streaming effect
   packet_placeholder = st.empty()
   
   for i in range(len(filtered_samples)):
       packet_placeholder.text(f"Packet {i+1}/{len(filtered_samples)}: {sample_data}")
       time.sleep(0.1)  # Simulate streaming delay
   ```

3. **Live Alert Ticker with Confidence Score**
   - Show predicted class + confidence bar
   - Color-coded: Green (Normal), Red (Attack)
   
   ```python
   confidence = model.predict_proba(sample)[0].max()
   st.progress(confidence)
   
   if prediction != 'Normal':
       st.error(f"🚨 ALERT: {prediction} detected (Confidence: {confidence:.2%})")
   else:
       st.success(f"✅ Normal traffic (Confidence: {confidence:.2%})")
   ```

4. **Attack Info Card** (What it Does, Real Example, Key Features)
   ```python
   # Load attack descriptions
   with open('data/attack_info.json', 'r') as f:
       attack_info = json.load(f)
   
   st.info(f"**What it does**: {attack_info[selected_attack]['description']}")
   st.code(attack_info[selected_attack]['example'])
   st.write(f"**Key Features**: {attack_info[selected_attack]['features']}")
   ```

5. **Speed Control Slider** (Packets/sec: 1-10)
   ```python
   speed = st.slider("Packets per second", min_value=1, max_value=10, value=5)
   delay = 1 / speed
   ```

6. **Alert Log Table** (Last 20 Detections)
   - Table showing: Timestamp, Attack Type, Confidence, Alert Status
   - Use `st.dataframe()` with auto-refresh

#### Code Structure:
```python
import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="Live Simulation", layout="wide")

# Load test samples
test_samples = pd.read_csv('data/test_samples.csv')

# Attack selector
attack_types = test_samples['attack_type'].unique()
selected_attack = st.selectbox("Select Attack Type", attack_types)

# Filter samples
filtered = test_samples[test_samples['attack_type'] == selected_attack]

# Animate
for idx, row in filtered.iterrows():
    # Predict
    prediction = model.predict([row.drop('attack_type')])
    confidence = model.predict_proba([row.drop('attack_type')])[0].max()
    
    # Display alert
    st.write(f"Packet {idx}: {prediction} ({confidence:.2%})")
    time.sleep(delay)
```

---

### **PAGE 3: EXPLAIN PREDICTION**
**File**: `pages/3_explain_prediction.py`

#### Features:
1. **Pick an Attack Type Selector** (Dropdown)
   - Select from 10 classes
   - Show raw feature values for that sample

2. **Model Prediction + Confidence Bar**
   ```python
   prediction = model.predict([sample])[0]
   proba = model.predict_proba([sample])[0]
   
   st.write(f"Predicted: **{label_encoder.inverse_transform([prediction])[0]}**")
   st.progress(proba.max())
   ```

3. **SHAP Waterfall Chart** (Why This Prediction?)
   ```python
   import shap
   
   # Load SHAP values
   with open('data/shap_values.pkl', 'rb') as f:
       shap_data = pickle.load(f)
   
   # Get SHAP values for selected sample
   sample_idx = test_samples[test_samples['attack_type'] == selected_attack].index[0]
   shap_values_sample = shap_data['shap_values'][sample_idx]
   
   # Waterfall plot
   fig = shap.plots.waterfall(shap_values_sample, show=False)
   st.pyplot(fig)
   ```

4. **Top 10 Features Driving Decision**
   - Bar chart of top features by |SHAP value|
   - Show actual feature values

5. **Plain-English Explanation** (Auto-generated)
   ```python
   # Get top 3 features
   top_features = feature_importance[:3]
   
   explanation = f"""
   This traffic was classified as **{prediction}** because:
   1. {top_features[0]['name']} was {top_features[0]['value']} (expected: {top_features[0]['expected']})
   2. {top_features[1]['name']} was {top_features[1]['value']} (expected: {top_features[1]['expected']})
   3. {top_features[2]['name']} was {top_features[2]['value']} (expected: {top_features[2]['expected']})
   """
   
   st.info(explanation)
   ```

#### Code Structure:
```python
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Explain Prediction", layout="wide")

# Load data
test_samples = pd.read_csv('data/test_samples.csv')
with open('data/shap_values.pkl', 'rb') as f:
    shap_data = pickle.load(f)

# Attack selector
selected_attack = st.selectbox("Pick Attack Type", test_samples['attack_type'].unique())

# Get sample
sample_idx = test_samples[test_samples['attack_type'] == selected_attack].index[0]
sample = test_samples.iloc[sample_idx].drop('attack_type')

# Predict
prediction = model.predict([sample])[0]
proba = model.predict_proba([sample])[0]

st.write(f"**Prediction**: {label_encoder.inverse_transform([prediction])[0]}")
st.progress(proba.max())

# SHAP waterfall
shap_values_sample = shap_data['shap_values'][sample_idx]
fig = shap.plots.waterfall(shap_values_sample, show=False)
st.pyplot(fig)

# Top features
top_features = get_top_features(shap_values_sample, sample)
st.bar_chart(top_features)

# Explanation
st.info(generate_explanation(prediction, top_features))
```

---

### **MAIN APP FILE**
**File**: `app.py`

```python
import streamlit as st

st.set_page_config(
    page_title="IDS ML Project",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Intrusion Detection System")
st.markdown("**UNSW-NB15 Dataset | Multi-Class Classification**")

st.sidebar.success("Select a page above")

st.write("""
## Welcome to the IDS ML Project

This application demonstrates a complete machine learning pipeline for network intrusion detection.

**Features:**
- 📊 **Dashboard**: Model comparison and metrics
- 🎯 **Live Simulation**: Real-time attack detection
- 🔍 **Explain Prediction**: SHAP-based interpretability

**Models Trained:**
- Logistic Regression (Baseline)
- Random Forest (200 trees)
- XGBoost (300 rounds)
- LightGBM (500 leaves) ⭐ Best Model

Navigate using the sidebar to explore!
""")
```

---

### **REQUIREMENTS.TXT**
```txt
streamlit==1.31.0
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
lightgbm==4.1.0
xgboost==2.0.3
shap==0.44.0
plotly==5.18.0
matplotlib==3.8.2
seaborn==0.13.0
```

---

### **DEPLOYMENT**

#### **Local Testing**
```bash
streamlit run app.py
```

#### **Streamlit Cloud Deployment**
1. Push code to GitHub repo
2. Go to https://share.streamlit.io
3. Connect repo and set `app.py` as entry point
4. Deploy!

---

## FINAL DELIVERABLES CHECKLIST

### **From Colab (Day 1-2)**
- ✅ `unsw_nb15.parquet` - Optimized dataset
- ✅ `feature_names.pkl` - Selected 30 features
- ✅ `label_encoder.pkl` - Class encoder
- ✅ `scaler.pkl` - StandardScaler
- ✅ `model_lgbm.pkl` - Best model
- ✅ `shap_values.pkl` - SHAP values (450 samples)
- ✅ `test_samples.csv` - Test samples with labels
- ✅ `attack_info.json` - Attack descriptions
- ✅ `metrics.json` - All model metrics

### **From Streamlit App**
- ✅ Page 1: Dashboard with 6 charts
- ✅ Page 2: Live simulation with streaming
- ✅ Page 3: SHAP explainability

---

## SUCCESS CRITERIA

1. **Data Pipeline**: CSV → Parquet conversion with 3x speedup ✅
2. **Data Quality**: <5% missing values after cleaning ✅
3. **Model Performance**: 
   - Accuracy > 85% ✅
   - Macro F1 > 0.80 ✅
   - Per-class F1 > 0.70 for all attack types ✅
4. **Explainability**: SHAP values computed for 50 samples per class ✅
5. **UI Functionality**: All 3 pages render without errors ✅

---

## TIMELINE SUMMARY

| Phase | Duration | Tasks | Tag |
|-------|----------|-------|-----|
| **Phase 1** | 30 min | Environment setup | `Setup` |
| **Phase 2** | 15 min | CSV → Parquet | `Storage` |
| **Phase 3** | 30 min | Data cleaning | `Quality` |
| **Phase 4** | 45 min | EDA + 4 charts | `EDA` |
| **Phase 5** | 30 min | Feature engineering | `Features` |
| **Phase 6** | 20 min | Preprocessing & split | `Prep` |
| **Phase 7** | 60 min | Train 4 models | `Training` |
| **Phase 8** | 45 min | Evaluation + metrics | `Results` |
| **Phase 9** | 30 min | SHAP analysis | `Export` |
| **Phase 10** | 120 min | Streamlit UI (3 pages) | `Streamlit` |

**Total Estimated Time**: 6-7 hours across 2 days

---

## NOTES FOR AI AGENT

### Critical Implementation Points:
1. **Always use `random_state=42`** for reproducibility
2. **Save all artifacts to Google Drive** immediately after creation
3. **Use stratified splitting** to preserve class distribution
4. **Handle class imbalance** with `class_weight='balanced'`
5. **Compress Parquet with Snappy** for optimal performance
6. **Test Streamlit locally** before deployment
7. **Document all hyperparameters** in code comments
8. **Log all metrics** to JSON for easy comparison

### Common Pitfalls to Avoid:
- ❌ Don't forget to strip whitespace from column names
- ❌ Don't mix up train/test scaling (fit on train only!)
- ❌ Don't use default class weights (data is imbalanced!)
- ❌ Don't skip saving the scaler and label encoder
- ❌ Don't hardcode file paths (use variables)

### Quality Checks:
- Verify Parquet file loads correctly
- Check for data leakage in preprocessing
- Validate stratification worked (check class distribution in splits)
- Test SHAP values on sample before full run
- Test Streamlit pages individually before integration

---

## APPENDIX: EXPECTED MODEL PERFORMANCE

Based on UNSW-NB15 benchmarks:

| Model | Expected Accuracy | Expected Macro F1 | Expected ROC-AUC |
|-------|------------------|-------------------|------------------|
| Logistic Regression | 75-80% | 0.70-0.75 | 0.85-0.88 |
| Random Forest | 85-88% | 0.80-0.85 | 0.92-0.95 |
| XGBoost | 88-91% | 0.85-0.88 | 0.94-0.96 |
| **LightGBM** | **90-93%** | **0.87-0.90** | **0.95-0.97** |

---

**END OF IMPLEMENTATION PLAN**
