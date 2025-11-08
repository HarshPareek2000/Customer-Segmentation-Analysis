import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class CustomerDataGenerator:
    """
    Generates realistic customer data for segmentation analysis.
    Creates 10,000+ customer records with various attributes.
    """
    
    def __init__(self, num_customers=10000):
        self.num_customers = num_customers
        self.regions = ['North', 'South', 'East', 'West', 'Central']
        self.product_categories = [
            'Electronics', 'Fashion', 'Home & Garden', 'Sports', 
            'Books', 'Groceries', 'Toys', 'Beauty', 'Automotive'
        ]
        self.genders = ['Male', 'Female']
        
    def generate_customers(self):
        """Generate customer data with realistic distributions."""
        
        np.random.seed(42)
        random.seed(42)
        
        print(f"Generating {self.num_customers} customer records...")
        
        # Generate customer IDs
        customer_ids = [f"CUST{str(i+1).zfill(6)}" for i in range(self.num_customers)]
        
        # Generate ages with realistic distribution
        ages = np.random.normal(45, 15, self.num_customers)
        ages = np.clip(ages, 18, 80).astype(int)
        
        # Generate genders
        genders = np.random.choice(self.genders, self.num_customers)
        
        # Generate annual income with realistic distribution
        incomes = np.random.gamma(2, 30000, self.num_customers)
        incomes = np.clip(incomes, 20000, 150000).astype(int)
        
        # Generate spending scores (correlated with income)
        spending_scores = []
        for income in incomes:
            # Higher income tends to have higher spending score
            base_score = (income - 20000) / (150000 - 20000) * 60
            score = np.random.normal(base_score + 20, 15)
            spending_scores.append(int(np.clip(score, 1, 100)))
        
        # Generate purchase frequency
        purchase_frequency = np.random.poisson(12, self.num_customers)
        purchase_frequency = np.clip(purchase_frequency, 1, 50)
        
        # Generate average transaction value (correlated with income and spending)
        avg_transaction_values = []
        for income, score in zip(incomes, spending_scores):
            base_value = income * 0.003 * (score / 50)
            value = np.random.normal(base_value, base_value * 0.3)
            avg_transaction_values.append(round(max(value, 10), 2))
        
        # Generate customer tenure (months)
        tenure_months = np.random.exponential(24, self.num_customers)
        tenure_months = np.clip(tenure_months, 1, 120).astype(int)
        
        # Generate product categories
        product_categories = np.random.choice(
            self.product_categories, 
            self.num_customers,
            p=[0.20, 0.15, 0.12, 0.10, 0.08, 0.10, 0.08, 0.09, 0.08]
        )
        
        # Generate regions
        regions = np.random.choice(self.regions, self.num_customers)
        
        # Create DataFrame
        df = pd.DataFrame({
            'customer_id': customer_ids,
            'age': ages,
            'gender': genders,
            'annual_income': incomes,
            'spending_score': spending_scores,
            'purchase_frequency': purchase_frequency,
            'avg_transaction_value': avg_transaction_values,
            'tenure_months': tenure_months,
            'product_category': product_categories,
            'region': regions
        })
        
        # Add some missing values (realistic scenario)
        missing_indices = np.random.choice(
            df.index, 
            size=int(self.num_customers * 0.02), 
            replace=False
        )
        df.loc[missing_indices, 'avg_transaction_value'] = np.nan
        
        print(f"✓ Generated {len(df)} customer records")
        print(f"✓ Age range: {df['age'].min()} - {df['age'].max()} years")
        print(f"✓ Income range: ${df['annual_income'].min():,} - ${df['annual_income'].max():,}")
        print(f"✓ Spending score range: {df['spending_score'].min()} - {df['spending_score'].max()}")
        print(f"✓ Missing values: {df.isnull().sum().sum()} cells")
        
        return df
    
    def save_to_csv(self, df, filename='data/customer_data.csv'):
        """Save generated data to CSV file."""
        
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        df.to_csv(filename, index=False)
        print(f"\n✓ Data saved to {filename}")
        print(f"\nDataset Info:")
        print(f"  - Total Customers: {len(df):,}")
        print(f"  - Features: {len(df.columns)}")
        print(f"  - File size: ~{os.path.getsize(filename) / 1024:.1f} KB")

def main():
    """
    Main function to generate customer data.
    """
    print("="*60)
    print("Customer Data Generator for Segmentation Analysis")
    print("="*60)
    print()
    
    # Generate data
    generator = CustomerDataGenerator(num_customers=10000)
    customer_df = generator.generate_customers()
    
    # Save to CSV
    generator.save_to_csv(customer_df)
    
    # Display sample
    print("\nSample Data (first 5 rows):")
    print(customer_df.head())
    
    print("\n" + "="*60)
    print("Data generation complete!")
    print("Run 'python scripts/clustering_analysis.py' for segmentation.")
    print("="*60)

if __name__ == "__main__":
    main()
