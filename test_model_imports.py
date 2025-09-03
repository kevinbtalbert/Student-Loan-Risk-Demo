#!/usr/bin/env python3
"""
Simple test script to validate model imports and initialization in CML environment.

Run this in CML before deployment to check if model_api.py will work correctly.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_imports():
    """Test if we can import and initialize the model API."""
    
    print("🔍 Testing Model Import and Initialization")
    print("=" * 50)
    
    try:
        # Test importing the model API
        logger.info("Importing model_api...")
        import model_api
        logger.info("✅ model_api imported successfully")
        
        # Test initialization
        logger.info("Testing model initialization...")
        init_result = model_api.init()
        
        if init_result:
            logger.info("✅ Model initialization successful!")
            
            # Test a simple prediction with correct schema
            logger.info("Testing prediction function...")
            test_data = {
                'borrower_id': 'TEST_001',
                'age': 28,
                'credit_score_at_origination': 720,
                'annual_income': 55000.0,
                'total_loan_amount': 45000.0,
                'loan_count': 2,
                'total_monthly_payment': 380.0
            }
            
            result = model_api.predict(test_data)
            logger.info(f"✅ Prediction test successful: {result}")
            
            print("\n🎉 All tests passed! Model should deploy successfully.")
            return True
            
        else:
            logger.error("❌ Model initialization failed")
            print("\n⚠️  Model initialization failed. Check the logs above for details.")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Import error: {str(e)}")
        print("\n⚠️  Import error. Make sure all dependencies are available.")
        return False
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        print(f"\n⚠️  Error during testing: {str(e)}")
        return False

def check_environment():
    """Check the CML environment setup."""
    
    print("\n🔧 CML Environment Check")
    print("=" * 30)
    
    # Check current directory and contents
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")
    if os.path.exists(current_dir):
        print(f"📄 Directory contents: {os.listdir(current_dir)}")
    
    # Check for required directories
    required_dirs = ['utils', 'models']
    for dirname in required_dirs:
        if os.path.exists(dirname):
            print(f"✅ {dirname}/ directory exists")
            if dirname == 'utils':
                print(f"   📄 Utils contents: {os.listdir(dirname)}")
            elif dirname == 'models':
                model_files = [f for f in os.listdir(dirname) if f.endswith('.joblib')]
                print(f"   📄 Model files: {model_files}")
        else:
            print(f"❌ {dirname}/ directory missing")
    
    # Check Python path
    print(f"\n🐍 Python path: {sys.path}")

if __name__ == "__main__":
    print("🚀 CML Model Testing Utility")
    print("=" * 40)
    
    # Check environment first
    check_environment()
    
    # Test model imports and initialization
    success = test_model_imports()
    
    if success:
        print("\n✅ Ready for deployment!")
        sys.exit(0)
    else:
        print("\n❌ Issues found. Please fix before deploying.")
        sys.exit(1)
