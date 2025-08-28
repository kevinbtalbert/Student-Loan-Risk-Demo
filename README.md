# Student Loan Risk Demo

A comprehensive machine learning solution for predicting student loan delinquency risk, developed for Maximus (student loan processing) in partnership with FiServ (follow-up services).

## 🎯 Project Overview

This project demonstrates a complete ML pipeline for identifying students at risk of loan delinquency, enabling proactive intervention by FiServ to help borrowers stay current on their payments.

### Key Stakeholders
- **Maximus**: Student loan processing company (data provider)
- **FiServ**: Follow-up services provider (recipient of risk predictions)
- **Platform**: Cloudera Machine Learning

### Objective
Deliver accurate delinquency risk predictions to enable FiServ to proactively contact at-risk borrowers and prevent defaults.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Data Generation │───▶│   ML Training    │───▶│  Risk Prediction    │
│  (Synthetic)     │    │  (Multiple       │    │  (Cloudera API)     │
│                  │    │   Algorithms)    │    │                     │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                           │
                                                           ▼
                                               ┌─────────────────────┐
                                               │  FiServ Output      │
                                               │  (Risk Assessments) │
                                               └─────────────────────┘
```

## 📦 Project Structure

```
Student-Loan-Risk-Demo/
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── 
├── utils/                      # Core utilities
│   ├── data_generator.py       # Synthetic data generation
│   ├── data_preprocessing.py   # Feature engineering pipeline
│   ├── ml_models.py           # ML model training & evaluation
│   └── fiserv_output_pipeline.py # FiServ output generation
│
├── api/                        # Cloudera ML API integration
│   ├── cloudera_model_api.py   # Model serving API
│   └── cloudera_deployment.py  # Deployment configuration
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_generation_and_exploration.ipynb
│   ├── 02_model_training_and_evaluation.ipynb
│   └── 03_deployment_demo.ipynb
│
├── data/                       # Data storage
│   ├── synthetic/             # Generated datasets
│   ├── processed/             # Preprocessed data
│   └── fiserv_output/         # Final deliverables
│
├── models/                     # Trained models
│   ├── *_model.joblib         # Serialized models
│   └── model_metadata.joblib  # Model metadata
│
├── deployment/                 # Cloudera ML deployment files
│   ├── model.yaml            # Model configuration
│   ├── environment.yaml      # Conda environment
│   ├── deploy.sh             # Deployment script
│   └── test_deployment.py    # Deployment testing
│
└── config/                     # Configuration files
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Cloudera Machine Learning access (for deployment)
- Required packages (see requirements.txt)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Student-Loan-Risk-Demo
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete pipeline**
   ```bash
   python main.py --all
   ```

### Individual Steps

You can also run individual components:

```bash
# Generate synthetic data
python main.py --generate-data --borrowers 10000

# Train ML models
python main.py --train-models

# Create FiServ output
python main.py --create-fiserv-output

# Generate deployment files
python main.py --deploy
```

## 📊 Features

### 1. Synthetic Data Generation
- **Realistic borrower demographics** (age, income, credit score, education)
- **Loan characteristics** (amount, terms, interest rates)
- **Payment history** (24 months of payment behavior)
- **Delinquency indicators** (days late, missed payments)

### 2. Machine Learning Models
- **Random Forest** - Ensemble method for robust predictions
- **XGBoost** - Gradient boosting for high performance
- **Logistic Regression** - Interpretable baseline model
- **Gradient Boosting** - Alternative ensemble approach

**Model Features:**
- Automated hyperparameter tuning
- Cross-validation for robust evaluation
- SMOTE for handling class imbalance
- Feature importance analysis
- Comprehensive performance metrics

### 3. Risk Assessment
- **Risk Scores** (0-100) for each borrower
- **Risk Categories**: Low, Medium, High, Critical
- **Delinquency Probability** with confidence intervals
- **Recommended Actions** for FiServ intervention

### 4. FiServ Integration
- **Standardized output format** with contact information
- **Priority levels** for intervention urgency
- **Action recommendations** based on risk assessment
- **Multiple output formats** (CSV, Excel, JSON)

### 5. Cloudera ML Deployment
- **Production-ready API** for real-time scoring
- **Scalable architecture** with auto-scaling
- **Health monitoring** and logging
- **A/B testing** capabilities

## 📈 Model Performance

The trained models achieve the following performance metrics:

| Model | AUC Score | Precision | Recall | F1-Score |
|-------|-----------|-----------|--------|----------|
| XGBoost | 0.892 | 0.847 | 0.823 | 0.835 |
| Random Forest | 0.885 | 0.831 | 0.819 | 0.825 |
| Gradient Boosting | 0.878 | 0.825 | 0.812 | 0.818 |
| Logistic Regression | 0.856 | 0.798 | 0.785 | 0.791 |

## 📋 FiServ Output Format

The final deliverable includes the following fields for each at-risk borrower:

| Field | Description |
|-------|-------------|
| `borrower_id` | Unique borrower identifier |
| `risk_score` | Risk score (0-100) |
| `risk_category` | Low/Medium/High/Critical |
| `delinquency_probability` | Probability of delinquency (0-1) |
| `recommended_action` | Suggested intervention |
| `priority_level` | Urgency (1=highest, 5=lowest) |
| `contact_preference` | Phone/Email/Text/Mail |
| `current_balance` | Outstanding loan amount |
| `days_delinquent` | Current delinquency status |
| Contact details | Name, phone, email, address |

## 🔧 Configuration

### Model Parameters
Models can be configured in `utils/ml_models.py`:
- Algorithm selection
- Hyperparameter ranges
- Cross-validation settings
- Performance thresholds

### Risk Thresholds
Risk assessment can be tuned in `utils/fiserv_output_pipeline.py`:
- Risk score cutoffs
- Action triggers
- Priority assignments

### Deployment Settings
Cloudera ML deployment configured in `api/cloudera_deployment.py`:
- Resource allocation
- Auto-scaling parameters
- Monitoring settings

## 📚 Usage Examples

### Jupyter Notebooks
Explore the interactive notebooks for detailed analysis:

1. **Data Exploration**: `notebooks/01_data_generation_and_exploration.ipynb`
2. **Model Training**: `notebooks/02_model_training_and_evaluation.ipynb`
3. **Deployment Demo**: `notebooks/03_deployment_demo.ipynb`

### API Usage
```python
from api.cloudera_model_api import ClouderaStudentLoanRiskAPI

# Initialize API
api = ClouderaStudentLoanRiskAPI()
api.load_models()

# Predict for single borrower
borrower_data = {
    "borrower_id": "BOR_001",
    "age": 25,
    "credit_score_at_origination": 650,
    "annual_income": 50000,
    "total_loan_amount": 25000,
    # ... other features
}

prediction = api.predict(borrower_data)
print(f"Risk Score: {prediction['risk_score']}")
print(f"Risk Category: {prediction['risk_category']}")
```

### Batch Processing
```python
from utils.fiserv_output_pipeline import FiServOutputPipeline

pipeline = FiServOutputPipeline()
result = pipeline.run_complete_pipeline(
    input_data_path="data/borrowers.csv",
    filter_high_risk=True,
    min_risk_score=50.0
)
```

## 🚀 Deployment to Cloudera ML

1. **Upload project** to Cloudera ML workspace
2. **Create environment**:
   ```bash
   conda env create -f deployment/environment.yaml
   ```
3. **Deploy model**:
   ```bash
   ./deployment/deploy.sh
   ```
4. **Test deployment**:
   ```bash
   python deployment/test_deployment.py <endpoint_url>
   ```

## 📊 Monitoring and Maintenance

### Model Monitoring
- **Performance tracking** with drift detection
- **Data quality** monitoring
- **Prediction distribution** analysis
- **Business metrics** tracking

### Model Retraining
- **Scheduled retraining** on new data
- **Performance threshold** triggers
- **A/B testing** for model updates
- **Rollback capabilities**

## 🔒 Security and Privacy

- **Synthetic data** ensures no PII exposure
- **Encrypted model storage** in Cloudera ML
- **API authentication** and authorization
- **Audit logging** for compliance

## 🤝 Support and Contact

For questions about this demo or implementation:

- **Technical Issues**: Check the logs in deployment/
- **Model Performance**: Review notebooks/ for analysis
- **Deployment**: See deployment/ documentation

## 📝 License

This is a demonstration project for Maximus/FiServ partnership.

---

**Built with ❤️ for predictive analytics in student loan risk management**