
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for premium aesthetic
sns.set_theme(style="whitegrid", context="talk", palette="viridis")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Lato', 'Arial', 'DejaVu Sans']

def load_data(filepath):
    """Loads the dataset from the CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def plot_churn_distribution(df, output_dir):
    """Plots the distribution of Churn vs Existing customers."""
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='Attrition_Flag', hue='Attrition_Flag', data=df, palette={'Existing Customer': '#2ecc71', 'Attrited Customer': '#e74c3c'}, legend=False)
    plt.title('Customer Attrition Distribution', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Customer Status', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.ylim(0, df.shape[0]*1.1)
    
    # Add annotations
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 10),
                    textcoords='offset points')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'churn_distribution.png'), dpi=300)
    plt.close()

def plot_heatmap(df, output_dir):
    """Plots a correlation heatmap for numerical features."""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Drop naive bayes columns if they exist (artifacts from kaggle)
    cols_to_drop = [c for c in numeric_df.columns if 'Naive_Bayes' in c or 'CLIENTNUM' in c]
    numeric_df = numeric_df.drop(columns=cols_to_drop)
    
    plt.figure(figsize=(16, 12))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', annot_kws={"size": 10})
    
    plt.title('Correlation Matrix of Numerical Features', fontsize=20, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300)
    plt.close()

def plot_income_category_churn(df, output_dir):
    """Plots Churn rate by Income Category."""
    plt.figure(figsize=(12, 8))
    
    # Calculate churn rate by income category
    # Ensure correct order of income categories if possible, otherwise rely on default
    income_order = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', 'Unknown']
    
    # Filter for only categories that exist in data to avoid errors
    existing_categories = [cat for cat in income_order if cat in df['Income_Category'].unique()]
    
    sns.countplot(x='Income_Category', hue='Attrition_Flag', data=df, order=existing_categories, palette='viridis')
    plt.title('Attrition by Income Category', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Income Category', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(title='Status')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'income_churn.png'), dpi=300)
    plt.close()

def plot_age_distribution(df, output_dir):
    """Plots age distribution of customers."""
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Customer_Age', hue='Attrition_Flag', kde=True, element="step", palette='viridis')
    plt.title('Customer Age Distribution by Attrition', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    import numpy as np # Import locally for heatmap mask
    
    DATA_PATH = 'BankChurners.csv' # Assuming in same dir
    OUTPUT_DIR = 'plots'
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Loading data...")
    try:
        df = load_data(DATA_PATH)
        print("Data loaded successfully.")
        
        print("Generating Churn Distribution plot...")
        plot_churn_distribution(df, OUTPUT_DIR)
        
        print("Generating Correlation Heatmap...")
        plot_heatmap(df, OUTPUT_DIR)
        
        print("Generating Income Category Churn plot...")
        plot_income_category_churn(df, OUTPUT_DIR)
        
        print("Generating Age Distribution plot...")
        plot_age_distribution(df, OUTPUT_DIR)
        
        print(f"All plots saved to {OUTPUT_DIR}")
        
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
