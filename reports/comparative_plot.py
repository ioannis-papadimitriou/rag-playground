import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ROOT_FOLDER = 'reports/qa_reports/'

def parse_results(text):
    # Add pattern to extract total QA pairs
    qa_pairs_pattern = r"Total QA pairs evaluated: (\d+)"
    
    # Updated metric pattern to capture std values
    overall_pattern = r"Evaluation Results Summary \((.*?)\).*?Passed QA pairs: \d+ \(([\d.]+)%\)\nOverall Score: ([\d.]+) ± ([\d.]+)"
    section_pattern = r"Evaluation Results Summary \((.*?)\).*?\nMetric\s+Mean\s+Std.*?Pass%.*?\n[-\s]+(.*?)(?=Evaluation Results Summary|\Z)"
    metric_pattern = r"(\w+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+\d+\.\d+\s+\d+\.\d+\s+(\d+\.\d+)"  # Updated to capture std
    
    # Parse total QA pairs for each model
    qa_pairs = {}
    for section in text.split("Evaluation Results Summary")[1:]:
        if match := re.search(qa_pairs_pattern, section):
            model_name = re.search(r"\((.*?)\)", section).group(1).split('/')[0].strip()
            qa_pairs[model_name] = int(match.group(1))
    
    # Parse overall metrics
    overall_data = []
    overall_matches = re.findall(overall_pattern, text, re.DOTALL)
    for model_name, pass_rate, score, std in overall_matches:
        model_name = model_name.split('/')[0].strip()
        overall_data.append({
            'name': model_name,
            'overall_score': float(score),
            'overall_std': float(std),
            'overall_pass_rate': float(pass_rate)/100,
            'total_qa_pairs': qa_pairs.get(model_name, 0)
        })
    
    # Parse detailed metrics
    metrics_data = []
    sections = re.findall(section_pattern, text, re.DOTALL)
    for model_name, section in sections:
        model_name = model_name.split('/')[0].strip()
        metrics = re.findall(metric_pattern, section.strip())
        
        for metric_name, mean, std, pass_rate in metrics:  # Updated to include std
            metrics_data.append({
                'name': model_name,
                'metric': metric_name,
                'mean': float(mean),
                'std': float(std),  # Now capturing std
                'pass_rate': float(pass_rate)/100,
                'total_qa_pairs': qa_pairs.get(model_name, 0)
            })
    
    # Add overall metrics to metrics_data
    for overall in overall_data:
        metrics_data.append({
            'name': overall['name'],
            'metric': 'overall',
            'mean': overall['overall_score'],
            'std': overall['overall_std'],
            'pass_rate': overall['overall_pass_rate'],
            'total_qa_pairs': overall['total_qa_pairs']
        })
    
    return pd.DataFrame(metrics_data), pd.DataFrame(overall_data)

def plot_metrics_comparison(metrics_df, overall_df, show_error_bars=False):
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
    })
    
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(2, 2, 
                         height_ratios=[1, 1],
                         width_ratios=[1.2, 1.2],
                         hspace=0.4,
                         wspace=0.3)
    
    ax_overall = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    
    n_models = len(metrics_df['name'].unique())
    colors = sns.color_palette("Set2", n_models)
    base_hatches = ['', '\\', '//', 'x', '+', 'o', 'O', '*', '.']
    hatches = [base_hatches[i % len(base_hatches)] for i in range(n_models)]
    
    def plot_overall_comparison(df, metrics_df, ax):
        x = np.arange(len(df))
        width = 0.2
        
        # Plot score bars
        score_bars = ax.bar(x - width, df['overall_score'], width, 
                          label='Overall Score', color='#4ECDC4', alpha=0.85)
        
        # Plot pass rate bars
        pass_bars = ax.bar(x, df['overall_pass_rate'], width,
                          label='Overall Pass Rate', color='#FF6B6B', alpha=0.85)
        
        # Add numerical accuracy bars
        num_acc_data = metrics_df[metrics_df['metric'] == 'numerical_accuracy']
        num_acc_score_bars = ax.bar(x + width, num_acc_data['mean'], width,
                                  label='Numerical Accuracy Score', color='#45B7D1', alpha=0.85)
        num_acc_pass_bars = ax.bar(x + 2*width, num_acc_data['pass_rate'], width,
                                  label='Numerical Accuracy Pass Rate', color='#FF9F1C', alpha=0.85)
        
        if show_error_bars:
            # Error bars for overall score
            ax.errorbar(x - width, df['overall_score'], yerr=df['overall_std'],
                       fmt='none', color='black', capsize=5)
            
            # Error bars for numerical accuracy
            num_acc_std = metrics_df[metrics_df['metric'] == 'numerical_accuracy']['std'].values
            ax.errorbar(x + width, num_acc_data['mean'], yerr=num_acc_std,
                       fmt='none', color='black', capsize=5)
            
            # For pass rates, calculate error using actual n for each model
            for i, (_, row) in enumerate(df.iterrows()):
                n = row['total_qa_pairs']
                
                # Overall pass rate error
                pass_rate_err = np.sqrt((row['overall_pass_rate'] * (1 - row['overall_pass_rate'])) / n)
                ax.errorbar(x[i], row['overall_pass_rate'], yerr=pass_rate_err,
                          fmt='none', color='black', capsize=5)
                
                # Numerical accuracy pass rate error
                num_acc_pass = num_acc_data[num_acc_data['name'] == row['name']]['pass_rate'].values[0]
                num_acc_pass_err = np.sqrt((num_acc_pass * (1 - num_acc_pass)) / n)
                ax.errorbar(x[i] + 2*width, num_acc_pass, yerr=num_acc_pass_err,
                          fmt='none', color='black', capsize=5)
        
        ax.set_ylabel('Score / Pass Rate')
        ax.set_title('Overall Performance and Numerical Accuracy')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(df['name'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # Add value labels
        for bars in [score_bars, pass_bars, num_acc_score_bars, num_acc_pass_bars]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
    
    def plot_metric_breakdown(df, ax, score_type):
        filtered_df = df[(df['metric'] != 'overall') & (df['metric'] != 'numerical_accuracy')]
        
        sns.barplot(
            data=filtered_df,
            x='metric',
            y=score_type,
            hue='name',
            ax=ax,
            palette=colors,
            alpha=0.85
        )
        
        for i, bars in enumerate(ax.containers):
            for bar in bars:
                if bar.get_height() > 0:
                    bar.set_hatch(hatches[i % len(hatches)])
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(title='Configurations', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Score' if score_type == 'mean' else 'Pass Rate')
        ax.set_title(f'{"Mean Scores" if score_type == "mean" else "Pass Rates"} by Metric')
    
    # Create plots
    plot_overall_comparison(overall_df, metrics_df, ax_overall)
    plot_metric_breakdown(metrics_df, ax1, 'mean')
    plot_metric_breakdown(metrics_df, ax2, 'pass_rate')
    
    fig.suptitle('Performance Analysis', fontsize=14, y=0.95)
    
    plt.savefig(
        ROOT_FOLDER + 'comparison.png',
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    plt.close()

# Run everything
with open(ROOT_FOLDER + 'general_report.txt', 'r') as f:
    general_text = f.read()

metrics_df, overall_df = parse_results(general_text)
plot_metrics_comparison(metrics_df, overall_df, show_error_bars=False)