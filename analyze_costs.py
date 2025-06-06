import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import scipy.stats as stats

class CostAnalysis:
    def __init__(self, output_dir, sample_data_only=False):
        self.output_dir = output_dir
        self.data = None
        self.sample_data_only = sample_data_only

    def load_data(self):
        if self.sample_data_only:
            print("Generating sample data only (500 records)...")
            self.data = self.generate_sample_data(500)
            print(f"Sample data generated: {len(self.data)} records")
            return
        print("Loading data from UNY Layout.xlsx...")
        df = pd.read_excel('UNY Layout.xlsx')
        # Rename columns to match expected names
        df = df.rename(columns={
            'ADMIT_DT': 'Admission_Date',
            'FACILITY': 'Provider_ID',
            'DRG Weight': 'DRG_Weight',
            'PAID_AMOUNT': 'Total_Paid',
            'CLAIM_TYPE': 'Patient_Type',
            'LEVEL2': 'Service_Type'
        })
        # Ensure date column is datetime
        df['Admission_Date'] = pd.to_datetime(df['Admission_Date'])
        # Calculate base rate
        df['Base_Rate'] = df['Total_Paid'] / df['DRG_Weight']
        # Add quarter column
        df['Quarter'] = pd.PeriodIndex(df['Admission_Date'], freq='Q').astype(str)
        self.data = df
        print(f"Loaded {len(self.data)} records")
        # Expand test data if real data has fewer than 100 records
        if len(self.data) < 100:
            print("Real data has fewer than 100 records. Expanding with synthetic data...")
            synthetic_data = self.generate_synthetic_data(100 - len(self.data))
            self.data = pd.concat([self.data, synthetic_data], ignore_index=True)
            print(f"Expanded data to {len(self.data)} records")

    def generate_sample_data(self, num_records):
        np.random.seed(42)
        providers = [f'Provider_{i}' for i in range(1, 8)]
        drgs = [f'DRG_{i}' for i in range(1, 6)]
        patient_types = ['IP', 'OP']
        service_types = ['Medical', 'Surgical', 'Psych', 'Rehab']
        # Ensure each provider has records in both Q4 2024 and Q1 2025
        records_per_provider = num_records // len(providers)
        data = []
        for provider in providers:
            # Generate Q4 2024 records
            dates_q4 = pd.date_range('2024-10-01', '2024-12-31', periods=records_per_provider//2)
            for date in dates_q4:
                data.append({
                    'Admission_Date': date,
                    'Provider_ID': provider,
                    'DRG': np.random.choice(drgs),
                    'DRG_Weight': np.round(np.random.uniform(0.5, 3.0), 2),
                    'Total_Paid': np.round(np.random.normal(15000, 5000), 2),
                    'Patient_Type': np.random.choice(patient_types),
                    'Service_Type': np.random.choice(service_types)
                })
            # Generate Q1 2025 records
            dates_q1 = pd.date_range('2025-01-01', '2025-03-31', periods=records_per_provider - records_per_provider//2)
            for date in dates_q1:
                data.append({
                    'Admission_Date': date,
                    'Provider_ID': provider,
                    'DRG': np.random.choice(drgs),
                    'DRG_Weight': np.round(np.random.uniform(0.5, 3.0), 2),
                    'Total_Paid': np.round(np.random.normal(15000, 5000), 2),
                    'Patient_Type': np.random.choice(patient_types),
                    'Service_Type': np.random.choice(service_types)
                })
        df = pd.DataFrame(data)
        df['Base_Rate'] = df['Total_Paid'] / df['DRG_Weight']
        df['Quarter'] = pd.PeriodIndex(df['Admission_Date'], freq='Q').astype(str)
        return df

    def generate_synthetic_data(self, num_records):
        """Generate synthetic data with realistic distributions based on real data."""
        np.random.seed(42)
        # Use real data's distributions for synthetic data
        real_providers = self.data['Provider_ID'].unique()
        real_drgs = self.data['DRG'].unique()
        real_patient_types = self.data['Patient_Type'].unique()
        real_service_types = self.data['Service_Type'].unique()
        # Generate synthetic records
        synthetic = {
            'Admission_Date': pd.date_range(start='2024-01-01', periods=num_records, freq='D'),
            'Provider_ID': np.random.choice(real_providers, num_records),
            'DRG': np.random.choice(real_drgs, num_records),
            'DRG_Weight': np.random.normal(self.data['DRG_Weight'].mean(), self.data['DRG_Weight'].std(), num_records).round(2),
            'Total_Paid': np.random.normal(self.data['Total_Paid'].mean(), self.data['Total_Paid'].std(), num_records).round(2),
            'Patient_Type': np.random.choice(real_patient_types, num_records),
            'Service_Type': np.random.choice(real_service_types, num_records)
        }
        synthetic_df = pd.DataFrame(synthetic)
        synthetic_df['Base_Rate'] = synthetic_df['Total_Paid'] / synthetic_df['DRG_Weight']
        synthetic_df['Quarter'] = pd.PeriodIndex(synthetic_df['Admission_Date'], freq='Q').astype(str)
        return synthetic_df

    def analyze_quarterly_costs(self):
        """Analyze costs by quarter"""
        # Calculate metrics by quarter and provider
        quarterly_metrics = self.data.groupby(['Quarter', 'Provider_ID']).agg({
            'Total_Paid': 'mean',
            'DRG_Weight': 'mean',
            'Base_Rate': 'mean'
        }).round(2)
        # Save results
        quarterly_metrics.to_csv(os.path.join(self.output_dir, 'quarterly_metrics.csv'))
        # Generate summary statistics
        summary_stats = self.data.groupby('Quarter').agg({
            'Total_Paid': ['mean', 'std', 'min', 'max', 'median'],
            'DRG_Weight': ['mean', 'std', 'median'],
            'Base_Rate': ['mean', 'std', 'median']
        }).round(2)
        summary_stats.to_csv(os.path.join(self.output_dir, 'summary_statistics.csv'))

    def analyze_top_providers(self):
        # Top 5 inpatient providers by volume
        top_providers = self.data[self.data['Patient_Type'] == 'IP']['Provider_ID'].value_counts().head(5).index
        top_provider_data = self.data[self.data['Provider_ID'].isin(top_providers)]
        provider_metrics = top_provider_data.groupby(['Provider_ID', 'Quarter']).agg({
            'Total_Paid': ['mean', 'median', 'std', 'count'],
            'DRG_Weight': ['mean', 'median'],
            'Base_Rate': ['mean', 'median']
        }).round(2)
        provider_metrics.to_csv(os.path.join(self.output_dir, 'top_providers_detailed.csv'))
        # Statistical tests Q4 2024 vs Q1 2025
        test_results = []
        for provider in top_providers:
            q4 = top_provider_data[(top_provider_data['Provider_ID'] == provider) & (top_provider_data['Quarter'] == '2024Q4')]['Total_Paid']
            q1 = top_provider_data[(top_provider_data['Provider_ID'] == provider) & (top_provider_data['Quarter'] == '2025Q1')]['Total_Paid']
            if len(q4) > 0 and len(q1) > 0:
                t_stat, p_val = stats.ttest_ind(q4, q1, equal_var=False)
                test_results.append({
                    'Provider_ID': provider,
                    'Q4_Mean': q4.mean(),
                    'Q1_Mean': q1.mean(),
                    'P_Value': p_val,
                    'Significant': p_val < 0.05
                })
        pd.DataFrame(test_results).to_csv(os.path.join(self.output_dir, 'top_providers_statistical_tests.csv'))

    def generate_visualizations(self):
        plt.style.use('seaborn-v0_8')
        # 1. Quarterly Cost Trends by Provider
        if len(self.data) > 0:
            plt.figure(figsize=(14, 7))
            sns.boxplot(data=self.data, x='Provider_ID', y='Total_Paid', hue='Quarter')
            plt.title('Quarterly Cost Distribution by Provider')
            plt.xlabel('Provider')
            plt.ylabel('Total Paid')
            plt.legend(title='Quarter')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'quarterly_cost_trends_by_provider.png'))
            plt.close()
        else:
            print("Warning: No data available for quarterly cost trends plot.")
        # 2. Provider Cost Distribution (Top 5) by Quarter
        top_providers = self.data[self.data['Patient_Type'] == 'IP']['Provider_ID'].value_counts().head(5).index
        if len(top_providers) > 0:
            plt.figure(figsize=(14, 7))
            sns.violinplot(data=self.data[self.data['Provider_ID'].isin(top_providers)], x='Provider_ID', y='Total_Paid', hue='Quarter', split=True)
            plt.title('Cost Distribution by Top Provider and Quarter')
            plt.xlabel('Provider')
            plt.ylabel('Total Paid')
            plt.legend(title='Quarter')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'provider_cost_violin_by_quarter.png'))
            plt.close()
        else:
            print("Warning: No top providers available for provider cost distribution plot.")
        # 3. DRG Weight vs Cost, colored by Provider
        if len(self.data) > 0:
            plt.figure(figsize=(12, 7))
            sns.scatterplot(data=self.data, x='DRG_Weight', y='Total_Paid', hue='Provider_ID', alpha=0.6)
            plt.title('DRG Weight vs Total Cost by Provider')
            plt.xlabel('DRG Weight')
            plt.ylabel('Total Paid')
            plt.legend(title='Provider', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'drg_weight_vs_cost_by_provider.png'))
            plt.close()
        else:
            print("Warning: No data available for DRG weight vs cost plot.")
        # 4. Q4 vs Q1 Comparison by Provider (Boxplot)
        q4_q1_data = self.data[self.data['Quarter'].isin(['2024Q4', '2025Q1'])]
        if len(q4_q1_data) > 0:
            plt.figure(figsize=(14, 7))
            sns.boxplot(data=q4_q1_data, x='Provider_ID', y='Total_Paid', hue='Quarter')
            plt.title('Q4 2024 vs Q1 2025 Cost Comparison by Provider')
            plt.xlabel('Provider')
            plt.ylabel('Total Paid')
            plt.legend(title='Quarter')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'q4_q1_comparison_by_provider.png'))
            plt.close()
        else:
            print("Warning: No data available for Q4 vs Q1 comparison plot.")
        # 5. Base Rate Trends by Provider
        if len(self.data) > 0:
            plt.figure(figsize=(14, 7))
            sns.boxplot(data=self.data, x='Provider_ID', y='Base_Rate', hue='Quarter')
            plt.title('Base Rate Trends by Provider and Quarter')
            plt.xlabel('Provider')
            plt.ylabel('Base Rate')
            plt.legend(title='Quarter')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'base_rate_trends_by_provider.png'))
            plt.close()
        else:
            print("Warning: No data available for base rate trends plot.")
        # 6. Service Type Costs by Provider
        if len(self.data) > 0:
            plt.figure(figsize=(14, 7))
            sns.boxplot(data=self.data, x='Provider_ID', y='Total_Paid', hue='Service_Type')
            plt.title('Cost Distribution by Provider and Service Type')
            plt.xlabel('Provider')
            plt.ylabel('Total Paid')
            plt.legend(title='Service Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'service_type_costs_by_provider.png'))
            plt.close()
        else:
            print("Warning: No data available for service type costs plot.")

    def generate_advanced_visualizations(self):
        plt.style.use('seaborn-v0_8')
        # 1. Violin Chart for Total_Paid by Provider and Quarter
        if len(self.data) > 0:
            plt.figure(figsize=(14, 7))
            sns.violinplot(data=self.data, x='Provider_ID', y='Total_Paid', hue='Quarter', split=True)
            plt.title('Violin Chart: Total Paid by Provider and Quarter')
            plt.xlabel('Provider')
            plt.ylabel('Total Paid')
            plt.legend(title='Quarter')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'violin_total_paid_by_provider_quarter.png'))
            plt.close()
        else:
            print("Warning: No data available for violin chart.")
        # 2. Heatmap for Provider vs Quarter (Mean Total_Paid)
        if len(self.data) > 0:
            pivot_data = self.data.pivot_table(index='Provider_ID', columns='Quarter', values='Total_Paid', aggfunc='mean')
            plt.figure(figsize=(14, 8))
            sns.heatmap(pivot_data, annot=True, cmap='YlGnBu')
            plt.title('Heatmap: Mean Total Paid by Provider and Quarter')
            plt.xlabel('Quarter')
            plt.ylabel('Provider')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'heatmap_provider_quarter.png'))
            plt.close()
        else:
            print("Warning: No data available for heatmap.")
        # 3. Line Plot for Mean and Median Total_Paid by Quarter for Each Provider
        if len(self.data) > 0:
            plt.figure(figsize=(14, 7))
            for provider, group in self.data.groupby('Provider_ID'):
                means = group.groupby('Quarter')['Total_Paid'].mean().reset_index()
                plt.plot(means['Quarter'].astype(str), means['Total_Paid'], marker='o', label=f'{provider} Mean')
            plt.title('Mean Total Paid by Quarter for Each Provider')
            plt.xlabel('Quarter')
            plt.ylabel('Mean Total Paid')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'line_plot_mean_by_provider.png'))
            plt.close()
            plt.figure(figsize=(14, 7))
            for provider, group in self.data.groupby('Provider_ID'):
                medians = group.groupby('Quarter')['Total_Paid'].median().reset_index()
                plt.plot(medians['Quarter'].astype(str), medians['Total_Paid'], marker='s', label=f'{provider} Median')
            plt.title('Median Total Paid by Quarter for Each Provider')
            plt.xlabel('Quarter')
            plt.ylabel('Median Total Paid')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'line_plot_median_by_provider.png'))
            plt.close()
        else:
            print("Warning: No data available for line plot.")
        # 4. Provider-Level Q1 vs Q4 Comparison (Bar, Normalized)
        q4_q1_data = self.data[self.data['Quarter'].isin(['2024Q4', '2025Q1'])]
        if len(q4_q1_data) > 0:
            provider_q4_q1 = q4_q1_data.groupby(['Provider_ID', 'Quarter'])['Total_Paid'].agg(['mean', 'median']).reset_index()
            pivot_mean = provider_q4_q1.pivot(index='Provider_ID', columns='Quarter', values='mean')
            pivot_median = provider_q4_q1.pivot(index='Provider_ID', columns='Quarter', values='median')
            plt.figure(figsize=(14, 7))
            pivot_mean.plot(kind='bar', ax=plt.gca())
            plt.title('Provider-Level Q4 vs Q1 Comparison (Mean)')
            plt.xlabel('Provider')
            plt.ylabel('Mean Total Paid')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'provider_q4_q1_mean.png'))
            plt.close()
            plt.figure(figsize=(14, 7))
            pivot_median.plot(kind='bar', ax=plt.gca())
            plt.title('Provider-Level Q4 vs Q1 Comparison (Median)')
            plt.xlabel('Provider')
            plt.ylabel('Median Total Paid')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'provider_q4_q1_median.png'))
            plt.close()
        else:
            print("Warning: No data available for provider-level Q4 vs Q1 comparison.")

    def generate_correlation_visuals(self):
        plt.style.use('seaborn-v0_8')
        # Filter data for Q4 2024 and Q1 2025
        q4_q1_data = self.data[self.data['Quarter'].isin(['2024Q4', '2025Q1'])]
        if len(q4_q1_data) == 0:
            print("Warning: No data available for Q4 vs Q1 correlation plot.")
            return
        # Group by Provider and Quarter, compute mean and median paid amounts, DRG Rate, and Base Rate
        provider_quarter_metrics = q4_q1_data.groupby(['Provider_ID', 'Quarter']).agg({
            'Total_Paid': ['mean', 'median'],
            'DRG_Weight': 'mean',
            'Base_Rate': 'mean'
        }).reset_index()
        # Rename columns for clarity
        provider_quarter_metrics.columns = ['Provider_ID', 'Quarter', 'Mean_Paid', 'Median_Paid', 'DRG_Rate', 'Base_Rate']
        # Create a scatter plot: Mean Paid vs DRG Rate, colored by Provider, with different markers for Q4 and Q1
        plt.figure(figsize=(14, 7))
        for provider, group in provider_quarter_metrics.groupby('Provider_ID'):
            q4 = group[group['Quarter'] == '2024Q4']
            q1 = group[group['Quarter'] == '2025Q1']
            if not q4.empty:
                plt.scatter(q4['DRG_Rate'], q4['Mean_Paid'], marker='o', label=f'{provider} Q4', alpha=0.7)
            if not q1.empty:
                plt.scatter(q1['DRG_Rate'], q1['Mean_Paid'], marker='s', label=f'{provider} Q1', alpha=0.7)
        plt.title('Mean Paid vs DRG Rate by Provider (Q4 vs Q1)')
        plt.xlabel('DRG Rate')
        plt.ylabel('Mean Paid')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mean_paid_vs_drg_rate_by_provider.png'))
        plt.close()
        # Create a scatter plot: Median Paid vs Base Rate, colored by Provider, with different markers for Q4 and Q1
        plt.figure(figsize=(14, 7))
        for provider, group in provider_quarter_metrics.groupby('Provider_ID'):
            q4 = group[group['Quarter'] == '2024Q4']
            q1 = group[group['Quarter'] == '2025Q1']
            if not q4.empty:
                plt.scatter(q4['Base_Rate'], q4['Median_Paid'], marker='o', label=f'{provider} Q4', alpha=0.7)
            if not q1.empty:
                plt.scatter(q1['Base_Rate'], q1['Median_Paid'], marker='s', label=f'{provider} Q1', alpha=0.7)
        plt.title('Median Paid vs Base Rate by Provider (Q4 vs Q1)')
        plt.xlabel('Base Rate')
        plt.ylabel('Median Paid')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'median_paid_vs_base_rate_by_provider.png'))
        plt.close()

    def run_analysis(self):
        """Run the complete analysis"""
        print("Loading data...")
        self.load_data()
        print("Analyzing quarterly costs...")
        self.analyze_quarterly_costs()
        print("Analyzing top providers...")
        self.analyze_top_providers()
        print("Generating visualizations...")
        self.generate_visualizations()
        print("Generating advanced visualizations...")
        self.generate_advanced_visualizations()
        print("Generating correlation visuals...")
        self.generate_correlation_visuals()
        print(f"Analysis complete. Results saved in {self.output_dir}")

if __name__ == "__main__":
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{timestamp}_Q1_2025_Cost_Analysis"
    os.makedirs(output_dir, exist_ok=True)
    # Set sample_data_only=True for this test
    analysis = CostAnalysis(output_dir, sample_data_only=True)
    analysis.run_analysis() 