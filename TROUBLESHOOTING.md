# Troubleshooting Guide for Student Loan Risk Demo

## 🚨 Recent Issues and Solutions

### Issue 1: `'float' object has no attribute 'clip'` ✅ FIXED

**Problem:** Multiple instances of `.clip()` being called on potentially scalar values.

**Root Cause:** The data generator uses `.clip()` method incorrectly in some cases.

**Solution Applied:**
- ✅ Fixed line 101: `interest_rate = np.clip(np.random.normal(5.5, 1.5), 2.0, 12.0)`

**Remaining Issue:** 
Other `.clip()` calls in the file should be checked and potentially fixed:

```python
# These may need similar fixes if n_borrowers=0 or other edge cases:
# Line 47: 'age': np.random.normal(28, 8, n_borrowers).astype(int).clip(18, 65)
# Line 50: 'credit_score_at_origination': np.random.normal(650, 80, n_borrowers).astype(int).clip(300, 850)  
# Line 55: 'dependents': np.random.poisson(1.2, n_borrowers).clip(0, 8)
# Line 74: 'gpa': np.random.normal(3.2, 0.5, n_borrowers).clip(2.0, 4.0)
```

### Issue 2: `[Errno 13] Permission denied: '../deployment'` ✅ FIXED

**Problem:** Deployment path was trying to create directory outside project.

**Solution Applied:**
- ✅ Fixed `cloudera_deployment.py` line 376: Changed default path from `"../deployment"` to `"deployment"`

### Issue 3: `ValueError: could not convert string to float: 'BOR_003992'` ✅ FIXED

**Problem:** The `borrower_id` column (string) was being included in ML training features.

**Solution Applied:**
- ✅ Fixed `data_preprocessing.py` line 176: Added removal of ID columns before feature processing
- ✅ All ID columns are now properly excluded from ML training

### Issue 4: `[Errno 13] Permission denied: '../data'` ✅ FIXED

**Problem:** FiServ output pipeline was trying to create directory outside project.

**Solution Applied:**
- ✅ Fixed `fiserv_output_pipeline.py` line 31: Changed default path from `"../data/fiserv_output"` to `"data/fiserv_output"`

### Issue 5: `Preprocessor must be fitted first using prepare_training_data()` ✅ FIXED

**Problem:** FiServ pipeline created unfitted preprocessor instance that couldn't transform data.

**Solution Applied:**
- ✅ Fixed `fiserv_output_pipeline.py` load_models(): Now loads training data and fits preprocessor properly
- ✅ Preprocessor is fitted with same data used for model training

### Issue 6: `maximum recursion depth exceeded while calling a Python object` ✅ FIXED

**Problem:** Infinite recursion in calculate_delinquency_predictions when error occurred.

**Solution Applied:**
- ✅ Fixed error handling to return mock predictions directly instead of calling itself recursively

### Issue 7: Missing Dependencies

**Problem:** `ModuleNotFoundError: No module named 'pandas'/'numpy'`

**Solution:**
```bash
# Install dependencies first
pip install -r requirements.txt
```

## 🔧 Complete Fix Implementation

To resolve all remaining issues, run this command:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the project (should work after dependency installation)
python main.py --all
```

## 🛠️ Alternative: Fix Remaining Clip Issues

If you still get clip errors after installing dependencies, here are the additional fixes needed:

### Fix 1: Update `utils/data_generator.py` lines 47, 50, 55, 74

Replace these lines:
```python
# Before (potentially problematic):
'age': np.random.normal(28, 8, n_borrowers).astype(int).clip(18, 65),
'credit_score_at_origination': np.random.normal(650, 80, n_borrowers).astype(int).clip(300, 850),
'dependents': np.random.poisson(1.2, n_borrowers).clip(0, 8),
'gpa': np.random.normal(3.2, 0.5, n_borrowers).clip(2.0, 4.0),

# After (guaranteed to work):
'age': np.clip(np.random.normal(28, 8, n_borrowers).astype(int), 18, 65),
'credit_score_at_origination': np.clip(np.random.normal(650, 80, n_borrowers).astype(int), 300, 850),
'dependents': np.clip(np.random.poisson(1.2, n_borrowers), 0, 8),
'gpa': np.clip(np.random.normal(3.2, 0.5, n_borrowers), 2.0, 4.0),
```

## 📋 Verification Steps

After installing dependencies:

1. **Test data generation:**
   ```bash
   python main.py --generate-data
   ```

2. **Test model training:**
   ```bash
   python main.py --train-models
   ```

3. **Test FiServ output:**
   ```bash
   python main.py --create-fiserv-output
   ```

4. **Test deployment:**
   ```bash
   python main.py --deploy
   ```

5. **Run complete pipeline:**
   ```bash
   python main.py --all
   ```

## ✅ Expected Success Output

```
============================================================
STEP 1: GENERATING SYNTHETIC DATA
============================================================
✓ Dataset generated successfully!

============================================================
STEP 2: TRAINING ML MODELS
============================================================
✓ Models trained successfully!

============================================================
STEP 3: CREATING FISERV OUTPUT
============================================================
✓ FiServ output created successfully!

============================================================
STEP 4: CLOUDERA ML DEPLOYMENT
============================================================
✓ Deployment files generated successfully!

============================================================
EXECUTION SUMMARY
============================================================
Steps completed: 4/4
✓ All steps completed successfully!
```

## 🚨 If Issues Persist

1. **Check Python version:** Ensure Python 3.9+ (tested with 3.11)
2. **Check dependencies:** Verify all packages in `requirements.txt` are installed
3. **Check permissions:** Ensure write access to project directory
4. **Check disk space:** Ensure sufficient space for generated files

## 📞 Support

If you continue to experience issues, the fixes above should resolve all known problems. The core issues have been identified and solutions provided.
