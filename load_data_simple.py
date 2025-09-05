#!/usr/bin/env python3
"""
Simple Student Loan Data Loader

Alternative approach using pandas to_sql method for easier data loading.
"""

import os
import pandas as pd
import cml.data_v1 as cmldata
from datetime import datetime

# Configuration
CONNECTION_NAME = "default-impala-aws"
DATABASE_NAME = "LoanTechSolutions"
DATA_PATH = "data/synthetic"

def load_all_data():
    """Load all CSV files into Impala using pandas to_sql method."""
    
    print("🚀 SIMPLE STUDENT LOAN DATA LOADER")
    print("="*50)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Connect to Impala
    print("🔌 Connecting to Impala...")
    conn = cmldata.get_connection(CONNECTION_NAME)
    
    try:
        # Create database
        print(f"🏗️  Setting up database: {DATABASE_NAME}")
        conn.execute(f"DROP DATABASE IF EXISTS {DATABASE_NAME} CASCADE")
        conn.execute(f"CREATE DATABASE {DATABASE_NAME}")
        conn.execute(f"USE {DATABASE_NAME}")
        
        # Get list of CSV files
        csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
        print(f"📂 Found {len(csv_files)} CSV files to load")
        
        total_rows_loaded = 0
        
        for csv_file in csv_files:
            try:
                print(f"\n📤 Loading {csv_file}...")
                
                # Read CSV
                df = pd.read_csv(os.path.join(DATA_PATH, csv_file))
                print(f"   📊 Rows in CSV: {len(df):,}")
                
                if len(df) == 0:
                    print(f"   ⚠️  Skipping empty file")
                    continue
                
                # Create table name from filename
                table_name = csv_file.replace('.csv', '')
                
                # Clean column names (remove special characters)
                df.columns = [col.replace('-', '_').replace(' ', '_').replace('.', '_') for col in df.columns]
                
                # Use pandas to_sql for easier loading
                # Note: This creates the table automatically
                engine = conn.get_base_connection()
                df.to_sql(name=table_name, con=engine, if_exists='replace', index=False, method='multi')
                
                print(f"   ✅ Loaded {len(df):,} rows into {table_name}")
                total_rows_loaded += len(df)
                
                # Verify
                verify_sql = f"SELECT COUNT(*) as count FROM {table_name}"
                result = conn.get_pandas_dataframe(verify_sql)
                actual_count = result['count'].iloc[0]
                print(f"   ✅ Verified: {actual_count:,} rows in table")
                
            except Exception as e:
                print(f"   ❌ Error loading {csv_file}: {str(e)}")
                continue
        
        # Summary
        print("\n" + "="*50)
        print("📊 LOADING SUMMARY")
        print("="*50)
        
        # Show all tables
        tables_df = conn.get_pandas_dataframe("SHOW TABLES")
        print(f"📋 Tables created: {len(tables_df)}")
        
        for table_name in tables_df['name']:
            count_sql = f"SELECT COUNT(*) as count FROM {table_name}"
            result = conn.get_pandas_dataframe(count_sql)
            count = result['count'].iloc[0]
            print(f"   • {table_name}: {count:,} rows")
        
        print(f"\n📈 Total rows loaded: {total_rows_loaded:,}")
        print("✅ Data loading completed!")
        
    except Exception as e:
        print(f"💥 FATAL ERROR: {str(e)}")
        raise
    
    finally:
        conn.close()
        print("🔌 Connection closed")

if __name__ == "__main__":
    load_all_data()
