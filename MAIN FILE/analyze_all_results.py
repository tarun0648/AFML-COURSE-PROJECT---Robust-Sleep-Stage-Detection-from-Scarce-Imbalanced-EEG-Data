"""
Comprehensive Results Analysis Script
Creates comparison tables for all experiments
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_scores(experiment_path):
    """Load scores.xlsx file"""
    scores_path = os.path.join(experiment_path, 'scores.xlsx')
    if os.path.exists(scores_path):
        df = pd.read_excel(scores_path)
        return df.iloc[0] if len(df) > 0 else None
    return None

def load_classification_report(experiment_path):
    """Load classification_report.xlsx file"""
    report_path = os.path.join(experiment_path, 'classification_report.xlsx')
    if os.path.exists(report_path):
        df = pd.read_excel(report_path)
        return df
    return None

def extract_metrics(experiment_dir):
    """Extract all metrics from an experiment directory"""
    metrics = {
        'experiment': None,
        'accuracy': None,
        'f1_score': None,
        'precision': None,
        'recall': None,
        'W_f1': None,
        'N1_f1': None,
        'N2_f1': None,
        'N3_f1': None,
        'REM_f1': None,
        'W_recall': None,
        'N1_recall': None,
        'N2_recall': None,
        'N3_recall': None,
        'REM_recall': None,
    }
    
    # Find all result directories
    base_path = Path(experiment_dir)
    
    # Look for scores.xlsx
    for scores_file in base_path.rglob('scores.xlsx'):
        scores_df = pd.read_excel(scores_file)
        if len(scores_df) > 0:
            acc_val = scores_df.iloc[0].get('acc', None)
            f1_val = scores_df.iloc[0].get('f1', None)
            
            # Handle if values are already in percentage (0-100) or decimal (0-1)
            if acc_val is not None:
                metrics['accuracy'] = acc_val / 100.0 if acc_val > 1.0 else acc_val
            if f1_val is not None:
                metrics['f1_score'] = f1_val / 100.0 if f1_val > 1.0 else f1_val
            break
    
    # Look for classification_report.xlsx
    for report_file in base_path.rglob('classification_report.xlsx'):
        report_df = pd.read_excel(report_file)
        
        # Check if format is: classes as columns (W, N1, N2, N3, REM), metrics as rows
        if 'W' in report_df.columns and 'N1' in report_df.columns:
            # Format: metrics are rows, classes are columns
            for idx, row in report_df.iterrows():
                metric_name = str(row.get('Unnamed: 0', '')).strip().lower()
                
                if 'f1' in metric_name or 'f1-score' in metric_name:
                    # Extract F1 scores for each class
                    if 'W' in report_df.columns:
                        val = row.get('W', None)
                        if val is not None:
                            metrics['W_f1'] = val / 100.0 if val > 1.0 else val
                    if 'N1' in report_df.columns:
                        val = row.get('N1', None)
                        if val is not None:
                            metrics['N1_f1'] = val / 100.0 if val > 1.0 else val
                    if 'N2' in report_df.columns:
                        val = row.get('N2', None)
                        if val is not None:
                            metrics['N2_f1'] = val / 100.0 if val > 1.0 else val
                    if 'N3' in report_df.columns:
                        val = row.get('N3', None)
                        if val is not None:
                            metrics['N3_f1'] = val / 100.0 if val > 1.0 else val
                    if 'REM' in report_df.columns:
                        val = row.get('REM', None)
                        if val is not None:
                            metrics['REM_f1'] = val / 100.0 if val > 1.0 else val
                    
                    # Also get macro avg F1
                    if 'macro avg' in report_df.columns:
                        val = row.get('macro avg', None)
                        if val is not None and metrics['f1_score'] is None:
                            metrics['f1_score'] = val / 100.0 if val > 1.0 else val
                
                elif 'recall' in metric_name:
                    # Extract recall for each class
                    if 'W' in report_df.columns:
                        val = row.get('W', None)
                        if val is not None:
                            metrics['W_recall'] = val / 100.0 if val > 1.0 else val
                    if 'N1' in report_df.columns:
                        val = row.get('N1', None)
                        if val is not None:
                            metrics['N1_recall'] = val / 100.0 if val > 1.0 else val
                    if 'N2' in report_df.columns:
                        val = row.get('N2', None)
                        if val is not None:
                            metrics['N2_recall'] = val / 100.0 if val > 1.0 else val
                    if 'N3' in report_df.columns:
                        val = row.get('N3', None)
                        if val is not None:
                            metrics['N3_recall'] = val / 100.0 if val > 1.0 else val
                    if 'REM' in report_df.columns:
                        val = row.get('REM', None)
                        if val is not None:
                            metrics['REM_recall'] = val / 100.0 if val > 1.0 else val
                    
                    # Also get macro avg recall
                    if 'macro avg' in report_df.columns:
                        val = row.get('macro avg', None)
                        if val is not None:
                            metrics['recall'] = val / 100.0 if val > 1.0 else val
                
                elif 'precision' in metric_name:
                    # Get macro avg precision
                    if 'macro avg' in report_df.columns:
                        val = row.get('macro avg', None)
                        if val is not None:
                            metrics['precision'] = val / 100.0 if val > 1.0 else val
        
        # Alternative format: classes as rows
        elif 'class' in report_df.columns or 'Class' in report_df.columns:
            class_col = 'class' if 'class' in report_df.columns else 'Class'
            
            for idx, row in report_df.iterrows():
                class_name = str(row.get(class_col, '')).strip()
                
                # Get values and normalize to 0-1 range
                f1_val = row.get('f1-score', row.get('f1', None))
                recall_val = row.get('recall', None)
                
                # Normalize if needed
                if f1_val is not None and f1_val > 1.0:
                    f1_val = f1_val / 100.0
                if recall_val is not None and recall_val > 1.0:
                    recall_val = recall_val / 100.0
                
                if 'W' in class_name or 'Wake' in class_name:
                    metrics['W_f1'] = f1_val
                    metrics['W_recall'] = recall_val
                elif 'N1' in class_name:
                    metrics['N1_f1'] = f1_val
                    metrics['N1_recall'] = recall_val
                elif 'N2' in class_name:
                    metrics['N2_f1'] = f1_val
                    metrics['N2_recall'] = recall_val
                elif 'N3' in class_name:
                    metrics['N3_f1'] = f1_val
                    metrics['N3_recall'] = recall_val
                elif 'REM' in class_name:
                    metrics['REM_f1'] = f1_val
                    metrics['REM_recall'] = recall_val
        
        break
    
    return metrics

def create_comparison_table():
    """Create comprehensive comparison table"""
    
    base_dir = 'experiments_logs_sleep_edf'
    
    experiments = [
        {
            'name': 'Baseline (No Augmentation)',
            'path': os.path.join(base_dir, 'Baseline_No_Aug', 'run_description', '_fold_0', 'supervised_100per'),
            'type': 'Baseline'
        },
        {
            'name': 'Baseline + VAE Augmentation',
            'path': os.path.join(base_dir, 'Baseline_VAE_Aug', 'run_description', '_fold_0', 'supervised_100per'),
            'type': 'VAE Augmented'
        },
        {
            'name': 'SSL SimCLR (Regular Data)',
            'path': os.path.join(base_dir, 'SSL_SimCLR', 'Pretrain', '_fold_0', 'ft_5per'),
            'type': 'SSL'
        },
        {
            'name': 'SSL SimCLR + VAE',
            'path': os.path.join(base_dir, 'SSL_SimCLR_VAE', 'Pretrain', '_fold_0', 'ft_5per'),
            'type': 'SSL + VAE'
        },
        # Alternative paths for SSL
        {
            'name': 'SSL SimCLR (Regular Data)',
            'path': os.path.join(base_dir, 'SSL_SimCLR', 'run_description', '_fold_0', 'ft_5per'),
            'type': 'SSL'
        },
        {
            'name': 'SSL SimCLR (Regular Data)',
            'path': os.path.join(base_dir, 'SSL_SimCLR', 'Finetune', '_fold_0', 'ft_5per'),
            'type': 'SSL'
        },
    ]
    
    all_metrics = []
    seen_experiments = set()  # Track which experiments we've already added
    
    for exp in experiments:
        exp_path = exp['path']
        exp_key = (exp['name'], exp['type'])  # Use name+type as unique key
        
        # Skip if we already have this experiment
        if exp_key in seen_experiments:
            continue
            
        if os.path.exists(exp_path):
            metrics = extract_metrics(exp_path)
            # Only add if we got some valid metrics
            if metrics.get('accuracy') is not None or metrics.get('f1_score') is not None:
                metrics['experiment'] = exp['name']
                metrics['type'] = exp['type']
                all_metrics.append(metrics)
                seen_experiments.add(exp_key)
            else:
                print(f"⚠️  Warning: {exp_path} exists but no metrics found")
        else:
            print(f"⚠️  Warning: {exp_path} not found")
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Reorder columns
    column_order = [
        'experiment', 'type', 'accuracy', 'f1_score', 'precision', 'recall',
        'W_f1', 'W_recall',
        'N1_f1', 'N1_recall',
        'N2_f1', 'N2_recall',
        'N3_f1', 'N3_recall',
        'REM_f1', 'REM_recall'
    ]
    
    # Only include columns that exist
    available_cols = [col for col in column_order if col in df.columns]
    df = df[available_cols]
    
    return df

def create_summary_table(df):
    """Create a summary table with key metrics"""
    
    summary_data = []
    
    for idx, row in df.iterrows():
        summary_data.append({
            'Method': row.get('experiment', 'Unknown'),
            'Type': row.get('type', 'Unknown'),
            'Accuracy (%)': f"{row.get('accuracy', 0)*100:.2f}" if row.get('accuracy') is not None else "N/A",
            'Macro F1 (%)': f"{row.get('f1_score', 0)*100:.2f}" if row.get('f1_score') is not None else "N/A",
            'N1 F1 (%)': f"{row.get('N1_f1', 0)*100:.2f}" if row.get('N1_f1') is not None else "N/A",
            'N1 Recall (%)': f"{row.get('N1_recall', 0)*100:.2f}" if row.get('N1_recall') is not None else "N/A",
            'N3 F1 (%)': f"{row.get('N3_f1', 0)*100:.2f}" if row.get('N3_f1') is not None else "N/A",
            'N3 Recall (%)': f"{row.get('N3_recall', 0)*100:.2f}" if row.get('N3_recall') is not None else "N/A",
        })
    
    return pd.DataFrame(summary_data)

def calculate_improvements(df):
    """Calculate improvements relative to baseline"""
    
    baseline_row = df[df['experiment'].str.contains('Baseline.*No Aug', case=False, na=False)]
    
    if len(baseline_row) == 0:
        return None
    
    baseline = baseline_row.iloc[0]
    improvements = []
    
    for idx, row in df.iterrows():
        if 'No Augmentation' in str(row.get('experiment', '')):
            continue
        
        imp = {
            'Method': row.get('experiment', 'Unknown'),
            'Accuracy Δ': None,
            'F1 Δ': None,
            'N1 F1 Δ': None,
            'N1 Recall Δ': None,
            'N3 F1 Δ': None,
            'N3 Recall Δ': None,
        }
        
        if baseline.get('accuracy') and row.get('accuracy'):
            imp['Accuracy Δ'] = f"{(row['accuracy'] - baseline['accuracy'])*100:+.2f}%"
        
        if baseline.get('f1_score') and row.get('f1_score'):
            imp['F1 Δ'] = f"{(row['f1_score'] - baseline['f1_score'])*100:+.2f}%"
        
        if baseline.get('N1_f1') and row.get('N1_f1'):
            imp['N1 F1 Δ'] = f"{(row['N1_f1'] - baseline['N1_f1'])*100:+.2f}%"
        
        if baseline.get('N1_recall') and row.get('N1_recall'):
            imp['N1 Recall Δ'] = f"{(row['N1_recall'] - baseline['N1_recall'])*100:+.2f}%"
        
        if baseline.get('N3_f1') and row.get('N3_f1'):
            imp['N3 F1 Δ'] = f"{(row['N3_f1'] - baseline['N3_f1'])*100:+.2f}%"
        
        if baseline.get('N3_recall') and row.get('N3_recall'):
            imp['N3 Recall Δ'] = f"{(row['N3_recall'] - baseline['N3_recall'])*100:+.2f}%"
        
        improvements.append(imp)
    
    return pd.DataFrame(improvements)

def main():
    print("="*80)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("="*80)
    print()
    
    # Create comparison table
    print("📊 Loading results from all experiments...")
    df = create_comparison_table()
    
    if df.empty:
        print("❌ No results found! Make sure experiments have completed.")
        return
    
    # Create summary table
    print("\n📈 Creating summary table...")
    summary_df = create_summary_table(df)
    
    # Calculate improvements
    print("📉 Calculating improvements vs baseline...")
    improvements_df = calculate_improvements(df)
    
    # Save to Excel
    output_file = 'experiments_logs_sleep_edf/COMPLETE_RESULTS_ANALYSIS.xlsx'
    print(f"\n💾 Saving results to {output_file}...")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        df.to_excel(writer, sheet_name='Detailed Results', index=False)
        if improvements_df is not None:
            improvements_df.to_excel(writer, sheet_name='Improvements vs Baseline', index=False)
    
    # Display results
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    if improvements_df is not None and not improvements_df.empty:
        print("\n" + "="*80)
        print("IMPROVEMENTS vs BASELINE")
        print("="*80)
        print(improvements_df.to_string(index=False))
    
    print("\n" + "="*80)
    print(f"✅ Analysis complete! Results saved to: {output_file}")
    print("="*80)
    
    # Print key findings
    print("\n🎯 KEY FINDINGS:")
    print("-" * 80)
    
    if len(df) >= 2:
        baseline = df[df['experiment'].str.contains('No Aug', case=False, na=False)]
        vae = df[df['experiment'].str.contains('VAE', case=False, na=False)]
        
        if len(baseline) > 0 and len(vae) > 0:
            baseline = baseline.iloc[0]
            vae = vae.iloc[0]
            
            if baseline.get('f1_score') and vae.get('f1_score'):
                f1_improvement = (vae['f1_score'] - baseline['f1_score']) * 100
                print(f"✓ VAE improves Macro F1 by {f1_improvement:+.2f}%")
            
            if baseline.get('N1_recall') and vae.get('N1_recall'):
                n1_improvement = (vae['N1_recall'] - baseline['N1_recall']) * 100
                print(f"✓ VAE improves N1 Recall by {n1_improvement:+.2f}%")
            
            if baseline.get('N3_recall') and vae.get('N3_recall'):
                n3_improvement = (vae['N3_recall'] - baseline['N3_recall']) * 100
                print(f"✓ VAE improves N3 Recall by {n3_improvement:+.2f}%")
    
    print()

if __name__ == "__main__":
    main()

