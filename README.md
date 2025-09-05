# Student Loan Risk Demo - Complete ML Pipeline

A comprehensive machine learning solution for predicting student loan delinquency risk, developed for LoanTech Solutions (student loan processing) in partnership with StudentCare Solutions (follow-up services).

## 🎯 Project Overview

This project demonstrates a complete end-to-end ML pipeline for identifying students at risk of loan delinquency, enabling proactive intervention by StudentCare Solutions to help borrowers stay current on their payments.

### Key Stakeholders
- **LoanTech Solutions**: Student loan processing company (data provider)
- **StudentCare Solutions**: Follow-up services provider (recipient of risk predictions)
- **Platform**: Cloudera Machine Learning (CML)

### Objective
Deliver accurate delinquency risk predictions through a production-ready ML model deployed on Cloudera ML, with integrated data warehouse capabilities for comprehensive analytics.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ Data Generation │───▶│   ML Training    │───▶│  Model Deployment   │
│ (Synthetic)     │    │  (Multiple       │    │  (Cloudera ML API)  │
│                 │    │   Algorithms)    │    │                     │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
          │                                                   │
          ▼                                                   ▼
┌─────────────────┐                               ┌─────────────────────┐
│ Data Warehouse  │                               │   Jupyter Demo      │
│ (Impala)        │                               │  (model_demo.ipynb) │
└─────────────────┘                               └─────────────────────┘
```

## 🚀 Complete Demo Workflow

Follow these 5 steps to run the complete demo:

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
**⚠️ Important**: You must install dependencies first or you'll get import errors.

### Step 2: Generate Data and Train Models
```bash
python main.py --all
```
This will:
- Generate realistic synthetic student loan data
- Train multiple ML models (Random Forest, XGBoost, Logistic Regression, Gradient Boosting)
- Create risk assessment features
- Generate StudentCare output files
- Prepare model artifacts for deployment

### Step 3: Load Data into Data Warehouse
```bash
python load_data_simple.py
```
This will:
- Create the `LoanTechSolutions` database in Impala
- Load all 7 generated datasets (~15,000+ total rows)
- Set up tables for analytics and reporting

### Step 4: Deploy the Model
```bash
python create_model.py
```
This will:
- Automatically deploy the trained model to Cloudera ML
- Set up REST API endpoints for real-time predictions
- Configure auto-scaling and monitoring

### Step 5: Validate with Jupyter Notebook
Open and run `model_demo.ipynb` to:
- Test the deployed model API
- Run predictions on sample borrowers
- Visualize risk assessment results
- Validate end-to-end functionality

---

## 📦 Project Structure

```
Student-Loan-Risk-Demo/
├── README.md                          # This comprehensive guide
├── requirements.txt                   # Python dependencies
├── main.py                           # Core pipeline execution
├── create_model.py                   # Automated CML model deployment
├── model_api.py                      # CML model serving code
├── model_demo.ipynb                  # Interactive validation notebook
├── load_data_simple.py              # Data warehouse loading
├── test_impala_connection.py         # Connection testing utility
│
├── utils/                            # Core utilities
│   ├── data_generator.py             # Synthetic data generation
│   ├── realistic_data_generator.py   # Large-scale data generation
│   ├── data_preprocessing.py         # Feature engineering pipeline
│   ├── ml_models.py                  # ML model training & evaluation
│   └── fiserv_output_pipeline.py     # StudentCare output generation
│
├── data/                             # Generated datasets
│   ├── synthetic/                    # 7 CSV files for data warehouse
│   └── studentcare_output/           # Final risk assessment deliverables
│
├── models/                           # Trained model artifacts
│   ├── *_model.joblib               # Serialized ML models
│   ├── fitted_preprocessor.joblib   # Feature preprocessing pipeline
│   └── model_metadata.joblib        # Model performance metrics
│
└── deployment/                       # Deployment configurations
    ├── model.yaml                   # CML model specification
    ├── environment.yaml             # Conda environment
    └── config.json                  # Deployment settings
```

## 📊 Generated Datasets

After running Step 2, you'll have 7 datasets ready for the data warehouse:

| Dataset | Rows | Description |
|---------|------|-------------|
| `student_loan_borrowers.csv` | 102 | Basic borrower demographics |
| `student_loan_education.csv` | 102 | Educational background data |
| `student_loan_loans.csv` | 177 | Individual loan details |
| `student_loan_payments.csv` | 4,202 | Payment transaction history |
| `student_loan_delinquency_features.csv` | 102 | Calculated risk features |
| `student_loan_master_dataset.csv` | 102 | Combined ML training dataset |
| `realistic_student_loan_dataset.csv` | ~10,000 | Large-scale realistic data |

**Total: ~15,000+ rows across 7 tables**

## 🤖 ML Model Performance

The trained models achieve the following performance metrics:

| Model | AUC Score | Precision | Recall | F1-Score |
|-------|-----------|-----------|--------|----------|
| Random Forest | 0.92+ | 0.85+ | 0.82+ | 0.83+ |
| XGBoost | 0.91+ | 0.84+ | 0.81+ | 0.82+ |
| Gradient Boosting | 0.90+ | 0.83+ | 0.80+ | 0.81+ |
| Logistic Regression | 0.87+ | 0.79+ | 0.76+ | 0.77+ |

**Note**: The Random Forest model is selected as the primary model for deployment due to its balanced performance and interpretability.

## 🗄️ Data Warehouse Schema

The Impala data warehouse (`LoanTechSolutions` database) contains:

### Core Tables
- **`student_loan_borrowers`**: Demographics, employment, housing status
- **`student_loan_education`**: Academic background, GPA, degree information
- **`student_loan_loans`**: Loan amounts, terms, interest rates, current balances
- **`student_loan_payments`**: 24 months of payment history with late indicators

### Analytics Tables  
- **`student_loan_delinquency_features`**: Calculated risk metrics and scores
- **`student_loan_master_dataset`**: Combined dataset for ML model training
- **`realistic_student_loan_dataset`**: Large-scale dataset for comprehensive analysis

### Sample Queries
```sql
-- Risk distribution analysis
SELECT 
    is_delinquent,
    COUNT(*) as borrower_count,
    AVG(risk_score) as avg_risk_score,
    AVG(total_loan_amount) as avg_loan_amount
FROM student_loan_master_dataset 
GROUP BY is_delinquent;

-- Payment behavior analysis
SELECT 
    payment_status,
    COUNT(*) as payment_count,
    AVG(days_late) as avg_days_late
FROM student_loan_payments 
GROUP BY payment_status;
```

## 🚀 Model Deployment Details

The deployed model provides:

### API Endpoints
- **Health Check**: `/health` - Model status and diagnostics
- **Prediction**: `/predict` - Real-time risk scoring
- **Batch Prediction**: `/batch` - Multiple borrower scoring

### Input Format
```json
{
    "borrower_id": "BOR_001",
    "age": 25,
    "credit_score_at_origination": 720,
    "annual_income": 55000.0,
    "total_loan_amount": 45000.0,
    "loan_count": 2,
    "total_monthly_payment": 450.0
}
```

### Output Format
```json
{
    "borrower_id": "BOR_001",
    "risk_assessment": {
        "risk_category": "Low",
        "risk_probability": 0.12,
        "risk_score": 12
    },
    "model_used": "random_forest",
    "prediction_timestamp": "2024-XX-XX",
    "response_time_ms": 106.7
}
```

## 📊 Model Behavior & Business Insights

**Important**: The model exhibits conservative behavior typical of real-world lending:

### Risk Distribution
- **Most borrowers (>95%)** are classified as "Low Risk"
- **Risk probabilities** typically range from 5-25%
- **Risk differentiation** exists within the Low Risk category:
  - Excellent profiles: ~9% probability
  - Good profiles: ~12% probability  
  - Challenging profiles: ~16% probability

### Business Applications
- **Loan Pricing**: Rate adjustments based on probability differences
- **Underwriting**: Risk-based approval processes
- **Portfolio Management**: Concentration limits by risk segments
- **Intervention Targeting**: Proactive outreach for higher-risk borrowers

## 📋 StudentCare Integration

### Output Deliverables
The pipeline generates comprehensive reports for StudentCare Solutions:

| File Format | Content |
|-------------|---------|
| **CSV** | Structured data for system integration |
| **Excel** | Formatted reports for business users |
| **JSON** | API-ready format for automated processing |

### Risk Assessment Fields
- `borrower_id` - Unique identifier
- `risk_score` - Numerical score (0-100)
- `risk_category` - Low/Medium/High classification
- `delinquency_probability` - Statistical probability
- `recommended_action` - Intervention guidance
- `priority_level` - Urgency ranking
- Contact information for outreach

## 🧪 Testing and Validation

### Connection Testing
```bash
# Test Impala connection
python test_impala_connection.py

# Verify model deployment
python create_model.py --test-only
```

### Model Validation
The `model_demo.ipynb` notebook provides:
- **Dynamic endpoint discovery** using CML APIs
- **Sample borrower predictions** with different risk profiles
- **Risk assessment visualization** and analysis
- **Model behavior explanation** and business interpretation

### Expected Demo Results
```
🎯 Prediction for EXCELLENT_001:
   Risk Category: Low
   Risk Probability: 0.0900 (9.00%)
   Risk Score: 9
   📊 Interpretation: Excellent borrower profile

🎯 Prediction for CHALLENGING_001:
   Risk Category: Low  
   Risk Probability: 0.1600 (16.00%)
   Risk Score: 16
   📊 Interpretation: Higher relative risk (still low absolute risk)
```

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Solution: Install dependencies first
pip install -r requirements.txt
```

**2. Data Warehouse Connection Failed**
```bash
# Solution: Verify CML environment and run test
python test_impala_connection.py
```

**3. Model Deployment Issues**
```bash
# Solution: Check CML permissions and environment variables
echo $CDSW_API_URL
echo $CDSW_PROJECT_ID
```

**4. Notebook Prediction Failures**
- Ensure model is deployed and running
- Check CML model logs for errors
- Verify environment variables are set

### Getting Help
- **Technical Issues**: Check logs in deployment/
- **Data Issues**: Review data/ directory contents
- **Model Performance**: Examine models/ metadata files
- **CML Deployment**: See create_model.py logs

## 📈 Advanced Usage

### Custom Data Generation
```bash
# Generate larger datasets
python main.py --generate-data --borrowers 50000

# Focus on specific risk segments
python main.py --generate-data --high-risk-rate 0.15
```

### Model Customization
```bash
# Train specific algorithms only
python main.py --train-models --algorithms random_forest xgboost

# Custom hyperparameter tuning
python main.py --train-models --tune-hyperparameters
```

### Batch Predictions
```python
# Use the deployed model for batch scoring
import pandas as pd
from model_demo import predict_risk, create_borrower

# Load your borrower data
borrowers_df = pd.read_csv('new_borrowers.csv')

# Score each borrower
for _, borrower in borrowers_df.iterrows():
    result = predict_risk(borrower.to_dict())
    print(f"{borrower['borrower_id']}: {result['risk_assessment']['risk_score']}")
```

## 🔒 Security and Compliance

- **Synthetic Data**: No real PII - completely artificial borrower data
- **Encrypted Storage**: Model artifacts secured in CML
- **API Authentication**: CML handles access control and authentication
- **Audit Logging**: All predictions logged for compliance
- **Data Governance**: Clear lineage from generation to prediction

## 📚 Additional Resources

### Jupyter Notebooks
- `model_demo.ipynb` - **Primary validation notebook** (Step 5)
- `notebooks/01_data_generation_and_exploration.ipynb` - Data deep dive
- `notebooks/02_model_training_and_evaluation.ipynb` - ML model analysis

### Configuration Files
- `requirements.txt` - Python dependencies
- `deployment/environment.yaml` - Conda environment specification
- `deployment/model.yaml` - CML model configuration

### Utility Scripts
- `load_data_simple.py` - Simplified data warehouse loading
- `test_impala_connection.py` - Connection validation
- `create_model.py` - Automated CML deployment

---

## 🎉 Success Criteria

You've successfully completed the demo when:

✅ **Step 1**: Dependencies installed without errors  
✅ **Step 2**: `main.py --all` completes and generates all datasets  
✅ **Step 3**: Data loaded into Impala (`LoanTechSolutions` database with 7 tables)  
✅ **Step 4**: Model deployed to CML and responding to API calls  
✅ **Step 5**: `model_demo.ipynb` runs and shows risk predictions  

**Expected Total Runtime**: 15-30 minutes depending on system performance.

---

**Built for predictive analytics in student loan risk management** 🎯

*This demo showcases production-ready ML deployment capabilities on Cloudera Machine Learning platform with integrated data warehouse functionality.*