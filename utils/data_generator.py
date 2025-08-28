"""
Synthetic Student Loan Data Generator for Risk Modeling Demo

This module generates realistic synthetic student loan data for demonstration purposes.
Data includes borrower demographics, loan characteristics, and historical payment patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Tuple, Dict, List


class StudentLoanDataGenerator:
    """Generate synthetic student loan data for risk modeling."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the data generator with a random seed for reproducibility."""
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Define categorical distributions
        self.schools = [
            'State University', 'Community College', 'Private University',
            'Technical Institute', 'Online University', 'Liberal Arts College',
            'Research University', 'Regional University'
        ]
        
        self.degree_types = ['Associates', 'Bachelors', 'Masters', 'Doctorate', 'Certificate']
        self.majors = [
            'Business', 'Engineering', 'Education', 'Healthcare', 'Liberal Arts',
            'Computer Science', 'Psychology', 'Biology', 'Art', 'Communications'
        ]
        
        self.employment_status = ['Employed Full-time', 'Employed Part-time', 'Unemployed', 'Student']
        self.states = [
            'CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
            'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI'
        ]
    
    def generate_borrower_demographics(self, n_borrowers: int) -> pd.DataFrame:
        """Generate borrower demographic information."""
        
        data = {
            'borrower_id': [f'BOR_{i:06d}' for i in range(1, n_borrowers + 1)],
            'age': np.random.normal(28, 8, n_borrowers).astype(int).clip(18, 65),
            'gender': np.random.choice(['M', 'F', 'O'], n_borrowers, p=[0.45, 0.52, 0.03]),
            'state': np.random.choice(self.states, n_borrowers),
            'credit_score_at_origination': np.random.normal(650, 80, n_borrowers).astype(int).clip(300, 850),
            'annual_income': np.random.lognormal(10.5, 0.6, n_borrowers).astype(int),
            'employment_status': np.random.choice(
                self.employment_status, n_borrowers, p=[0.65, 0.15, 0.12, 0.08]
            ),
            'dependents': np.random.poisson(1.2, n_borrowers).clip(0, 8),
            'housing_status': np.random.choice(
                ['Own', 'Rent', 'Family'], n_borrowers, p=[0.35, 0.55, 0.10]
            )
        }
        
        return pd.DataFrame(data)
    
    def generate_education_data(self, borrower_df: pd.DataFrame) -> pd.DataFrame:
        """Generate education-related information for borrowers."""
        
        n_borrowers = len(borrower_df)
        
        education_data = {
            'borrower_id': borrower_df['borrower_id'],
            'school_name': np.random.choice(self.schools, n_borrowers),
            'degree_type': np.random.choice(self.degree_types, n_borrowers),
            'major': np.random.choice(self.majors, n_borrowers),
            'graduation_year': np.random.randint(2010, 2024, n_borrowers),
            'gpa': np.random.normal(3.2, 0.5, n_borrowers).clip(2.0, 4.0),
            'school_type': np.random.choice(['Public', 'Private'], n_borrowers, p=[0.7, 0.3]),
            'completion_status': np.random.choice(
                ['Completed', 'Dropped Out', 'Transferred'], n_borrowers, p=[0.75, 0.15, 0.10]
            )
        }
        
        return pd.DataFrame(education_data)
    
    def generate_loan_data(self, borrower_df: pd.DataFrame) -> pd.DataFrame:
        """Generate loan characteristics for each borrower."""
        
        loans = []
        
        for _, borrower in borrower_df.iterrows():
            # Number of loans per borrower (1-4)
            n_loans = np.random.choice([1, 2, 3, 4], p=[0.4, 0.35, 0.2, 0.05])
            
            for loan_idx in range(n_loans):
                loan_amount = np.random.lognormal(9.5, 0.8) * 1000
                loan_amount = max(1000, min(loan_amount, 100000))  # Cap between $1K-$100K
                
                origination_date = datetime(2020, 1, 1) + timedelta(
                    days=np.random.randint(0, 1461)  # Random date in last 4 years
                )
                
                loan_term = np.random.choice([120, 240, 360], p=[0.2, 0.5, 0.3])  # 10, 20, 30 years
                interest_rate = np.clip(np.random.normal(5.5, 1.5), 2.0, 12.0)
                
                loans.append({
                    'loan_id': f'LOAN_{len(loans)+1:08d}',
                    'borrower_id': borrower['borrower_id'],
                    'loan_amount': round(loan_amount, 2),
                    'origination_date': origination_date,
                    'loan_term_months': loan_term,
                    'interest_rate': round(interest_rate, 3),
                    'loan_type': np.random.choice(['Subsidized', 'Unsubsidized', 'PLUS'], p=[0.4, 0.45, 0.15]),
                    'loan_status': 'Active',
                    'current_balance': round(loan_amount * np.random.uniform(0.5, 1.0), 2),
                    'monthly_payment': round((loan_amount * (interest_rate/100/12)) / 
                                           (1 - (1 + interest_rate/100/12)**(-loan_term)), 2)
                })
        
        return pd.DataFrame(loans)
    
    def generate_payment_history(self, loan_df: pd.DataFrame, months_history: int = 24) -> pd.DataFrame:
        """Generate payment history for each loan."""
        
        payment_records = []
        
        for _, loan in loan_df.iterrows():
            origination_date = pd.to_datetime(loan['origination_date'])
            monthly_payment = loan['monthly_payment']
            
            # Generate payment history for the specified number of months
            for month_offset in range(months_history):
                payment_date = origination_date + pd.DateOffset(months=month_offset)
                
                # Simulate payment behavior with some borrowers being more likely to miss payments
                payment_probability = 0.92  # 92% chance of making payment on time
                
                if np.random.random() < payment_probability:
                    payment_amount = monthly_payment
                    payment_status = 'On Time'
                    days_late = 0
                else:
                    # Late or missed payment
                    late_prob = np.random.random()
                    if late_prob < 0.6:  # 60% of late payments are 1-30 days late
                        days_late = np.random.randint(1, 31)
                        payment_amount = monthly_payment
                        payment_status = '1-30 Days Late'
                    elif late_prob < 0.8:  # 20% are 31-60 days late
                        days_late = np.random.randint(31, 61)
                        payment_amount = monthly_payment
                        payment_status = '31-60 Days Late'
                    else:  # 20% are missed payments
                        days_late = np.random.randint(61, 120)
                        payment_amount = 0
                        payment_status = 'Missed Payment'
                
                payment_records.append({
                    'payment_id': f'PAY_{len(payment_records)+1:08d}',
                    'loan_id': loan['loan_id'],
                    'borrower_id': loan['borrower_id'],
                    'payment_date': payment_date,
                    'scheduled_amount': monthly_payment,
                    'actual_amount': payment_amount,
                    'payment_status': payment_status,
                    'days_late': days_late
                })
        
        return pd.DataFrame(payment_records)
    
    def calculate_delinquency_features(self, payment_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate delinquency risk features from payment history."""
        
        # Aggregate payment statistics by borrower
        borrower_stats = payment_df.groupby('borrower_id').agg({
            'days_late': ['mean', 'max', 'std'],
            'actual_amount': ['sum', 'count'],
            'scheduled_amount': 'sum',
            'payment_status': lambda x: (x == 'Missed Payment').sum()
        }).round(2)
        
        # Flatten column names
        borrower_stats.columns = [
            'avg_days_late', 'max_days_late', 'std_days_late',
            'total_payments_made', 'total_payment_count', 'total_scheduled',
            'missed_payment_count'
        ]
        
        # Calculate additional features
        borrower_stats['payment_ratio'] = (
            borrower_stats['total_payments_made'] / borrower_stats['total_scheduled']
        ).fillna(0)
        
        borrower_stats['missed_payment_rate'] = (
            borrower_stats['missed_payment_count'] / borrower_stats['total_payment_count']
        ).fillna(0)
        
        # Calculate recent payment behavior (last 6 months)
        recent_payments = payment_df[
            payment_df['payment_date'] >= payment_df['payment_date'].max() - pd.DateOffset(months=6)
        ]
        
        recent_stats = recent_payments.groupby('borrower_id').agg({
            'days_late': 'mean',
            'payment_status': lambda x: (x == 'Missed Payment').sum()
        }).round(2)
        
        recent_stats.columns = ['recent_avg_days_late', 'recent_missed_payments']
        
        # Merge recent stats
        borrower_stats = borrower_stats.join(recent_stats, how='left').fillna(0)
        
        # Create delinquency target variable
        # Consider borrower delinquent if they have high recent missed payments or severe lateness
        borrower_stats['is_delinquent'] = (
            (borrower_stats['recent_missed_payments'] >= 2) |
            (borrower_stats['recent_avg_days_late'] > 30) |
            (borrower_stats['max_days_late'] > 90)
        ).astype(int)
        
        # Create risk score (0-100)
        borrower_stats['risk_score'] = (
            borrower_stats['missed_payment_rate'] * 40 +
            borrower_stats['recent_avg_days_late'] / 30 * 30 +
            (borrower_stats['max_days_late'] > 60).astype(int) * 30
        ).clip(0, 100).round(1)
        
        return borrower_stats.reset_index()
    
    def generate_complete_dataset(self, n_borrowers: int = 10000) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Generate a complete synthetic dataset with all components."""
        
        print(f"Generating synthetic student loan dataset with {n_borrowers} borrowers...")
        
        # Generate all components
        borrower_df = self.generate_borrower_demographics(n_borrowers)
        education_df = self.generate_education_data(borrower_df)
        loan_df = self.generate_loan_data(borrower_df)
        payment_df = self.generate_payment_history(loan_df)
        delinquency_df = self.calculate_delinquency_features(payment_df)
        
        # Create master dataset for ML modeling
        master_df = borrower_df.merge(education_df, on='borrower_id') \
                              .merge(delinquency_df, on='borrower_id')
        
        # Add loan summary statistics
        loan_summary = loan_df.groupby('borrower_id').agg({
            'loan_amount': ['sum', 'count', 'mean'],
            'interest_rate': 'mean',
            'current_balance': 'sum',
            'monthly_payment': 'sum'
        }).round(2)
        
        loan_summary.columns = [
            'total_loan_amount', 'loan_count', 'avg_loan_amount',
            'avg_interest_rate', 'total_current_balance', 'total_monthly_payment'
        ]
        
        master_df = master_df.merge(loan_summary.reset_index(), on='borrower_id')
        
        # Calculate debt-to-income ratio
        master_df['debt_to_income_ratio'] = (
            master_df['total_monthly_payment'] * 12 / master_df['annual_income']
        ).round(3)
        
        component_datasets = {
            'borrowers': borrower_df,
            'education': education_df,
            'loans': loan_df,
            'payments': payment_df,
            'delinquency_features': delinquency_df
        }
        
        print(f"Dataset generation complete!")
        print(f"- {len(borrower_df)} borrowers")
        print(f"- {len(loan_df)} loans")
        print(f"- {len(payment_df)} payment records")
        print(f"- Delinquency rate: {master_df['is_delinquent'].mean():.1%}")
        
        return master_df, component_datasets


def main():
    """Generate and save synthetic datasets."""
    generator = StudentLoanDataGenerator(random_seed=42)
    
    # Generate datasets
    master_df, component_datasets = generator.generate_complete_dataset(n_borrowers=10000)
    
    # Save datasets
    import os
    
    data_dir = '../data/synthetic'
    os.makedirs(data_dir, exist_ok=True)
    
    # Save master dataset
    master_df.to_csv(f'{data_dir}/student_loan_master_dataset.csv', index=False)
    
    # Save component datasets
    for name, df in component_datasets.items():
        df.to_csv(f'{data_dir}/student_loan_{name}.csv', index=False)
    
    print(f"\nDatasets saved to {data_dir}/")


if __name__ == "__main__":
    main()
