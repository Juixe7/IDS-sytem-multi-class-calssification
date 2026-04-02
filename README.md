# Intrusion Detection System (IDS) - ML Project

## 🛡️ Project Overview

This project implements a complete Machine Learning pipeline for network intrusion detection using the **UNSW-NB15 Dataset**. The system models network behavior to classify traffic into 10 categories (9 specific attack types and 1 normal traffic class), making it a comprehensive multi-class classification enterprise.

The project features a **Streamlit application** with interactive dashboards, live simulation of incoming network packets, and explainable AI capabilities using SHAP.

### Key Features
- **Multi-Class Classification**: Identifies 9 different types of attacks (e.g., Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms) and Normal traffic.
- **Model Comparison**: Four Machine Learning models were trained and evaluated:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM (Best performing model)
- **Interactive UI (Streamlit)**:
  - **📊 Dashboard**: View comparative metrics, confusion matrix, and ROC curves across all models.
  - **🎯 Live Simulation**: Simulate real-time packet streams to observe the live alert ticker, detection probabilities, and logging.
  - **🔍 Explainable AI**: Leverage SHAP values to dynamically understand which features most strongly influenced the model's predictions via waterfall charts and plain-English explanations.

## 📁 Repository Structure

```
.
├── src/                # Streamlit App Pages and utility functions
├── app.py              # Main Streamlit application entry point
├── main.py             # Backend Python logic or pipeline coordinator
├── test.py             # Primary testing scripts
├── test_explain.py     # Testing local SHAP explanation features
├── requirements.txt    # Python dependencies
└── .gitignore          # Ignored files (models, outputs, cache, etc.)
```

*(Note: Data, saved models, and artifacts are explicitly ignored in this repository to prevent large file sizes).*

## 🚀 Running the Project

### Prerequisites
Make sure you have Python 3.11 installed. It is highly recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

*(Note: In order to fully run everything, you will need the exported baseline models, scalers, encoders, and artifacts locally since they are not committed to Git).*

### Starting the Streamlit App
Run the following command in your terminal to start the dashboard:
```bash
streamlit run app.py
```

## 🛠️ Built With
- **Python Data Stack**: pandas, numpy, scikit-learn
- **Machine Learning**: LightGBM, XGBoost, Random Forest
- **Explainability**: SHAP
- **Frontend/UI**: Streamlit, Plotly, Matplotlib
- **Data Engineering**: PyArrow/Parquet (for chunked dataset reading & caching)

## 👤 Author
*GitHub user: [Juixe7](https://github.com/Juixe7)*
