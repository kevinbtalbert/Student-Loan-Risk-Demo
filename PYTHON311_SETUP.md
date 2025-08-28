# Python 3.11 Setup Guide

## 🐍 **Fixed for Python 3.11 Compatibility**

The project has been updated to work with Python 3.11. Here's what was changed and how to proceed:

## ✅ **What Was Fixed**

### 1. **Updated requirements.txt**
- Changed from exact versions (`==`) to minimum versions (`>=`)
- Removed problematic packages (`cloudml==1.0.0`, `cdsw==1.0.0`)
- Added comments explaining CML-specific packages

### 2. **Updated Deployment Configuration**
- Changed Python version from 3.9 to 3.11 in `api/cloudera_deployment.py`
- Updated conda package specifications
- Removed pip packages that aren't available on PyPI

### 3. **Enhanced Error Handling**
- Added graceful handling for missing `cdsw` module during local development
- Improved import error messages

## 🚀 **Installation Instructions**

### **For Local Development:**
```bash
# Use the updated requirements file
pip install -r requirements.txt
```

### **For CML Deployment:**
```bash
# Use the CML-specific requirements file
pip install -r requirements-cml.txt
```

## 📋 **Package Explanations**

| Package | Status | Notes |
|---------|--------|-------|
| `pandas>=2.1.0` | ✅ Compatible | Core data manipulation |
| `numpy>=1.24.0` | ✅ Compatible | Numerical computing |
| `scikit-learn>=1.3.0` | ✅ Compatible | Machine learning |
| `xgboost>=2.0.0` | ✅ Compatible | Gradient boosting |
| `cdsw` | ⚠️ CML Only | Pre-installed in CML environments |
| `cloudml` | ⚠️ CML Only | Pre-installed in CML environments |

## 🔧 **If You Still Get Errors**

### **Error: "Could not find a version that satisfies the requirement"**
```bash
# Solution: Use the flexible requirements
pip install -r requirements-cml.txt
```

### **Error: "ImportError: No module named 'cdsw'"**
```bash
# This is normal for local development
# The code will run in local mode without CML features
```

### **Error: "numpy version compatibility"**
```bash
# Update to latest compatible versions
pip install --upgrade numpy pandas scikit-learn
```

## 🧪 **Test Your Installation**

```bash
# Test the installation
python -c "
import pandas as pd
import numpy as np
import sklearn
import xgboost
print('✅ All core packages imported successfully!')
print(f'Python: {sys.version.split()[0]}')
print(f'Pandas: {pd.__version__}')
print(f'NumPy: {np.__version__}')
print(f'Scikit-learn: {sklearn.__version__}')
print(f'XGBoost: {xgboost.__version__}')
"
```

## 🚀 **Run the Project**

```bash
# Generate data and train models
python main.py --all

# Or run individual steps
python main.py --generate-data
python main.py --train-models
python main.py --create-studentcare-output
```

## 📝 **Files Updated for Python 3.11**

- ✅ `requirements.txt` - Updated package versions
- ✅ `requirements-cml.txt` - New CML-specific requirements
- ✅ `api/cloudera_deployment.py` - Updated Python version to 3.11
- ✅ `api/cloudera_model_api.py` - Added graceful CDSW import handling
- ✅ `README.md` - Updated Python version requirements

## 🎯 **Expected Results**

After running `pip install -r requirements.txt`, you should see:
```
Successfully installed pandas-2.1.4 numpy-1.24.3 scikit-learn-1.3.2 ...
✅ No errors about missing cloudml or cdsw packages
✅ All imports working correctly
```

The project is now fully compatible with Python 3.11! 🎉
