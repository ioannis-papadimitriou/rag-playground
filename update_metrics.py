import json
from pathlib import Path

# Define metric weights matching your MetricsEngine class
METRIC_WEIGHTS = {
    "key_terms_precision": 0.15,
    "token_recall": 0.15,
    "truthfulness": 0.2,
    "completeness": 0.1,
    "source_relevance": 0.05,
    "context_faithfulness": 0.1,
    "semantic_f1": 0.1,
    "completeness_gain": 0.05,
    "answer_relevance": 0.1
}

def update_evaluation_report(report_path: str):
    """Update existing evaluation report with overall metrics"""
    
    # Load existing report
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    # Calculate overall metrics for each result
    for result in report['results']:
        # Get all metrics scores and weights
        weighted_sum = 0
        total_weight = 0
        passed_metrics = 0
        total_metrics = len(result['metrics'])
        
        for metric in result['metrics']:
            metric_name = metric['name']
            weight = METRIC_WEIGHTS.get(metric_name, 0.1)  # Default weight 0.1 if not found
            weighted_sum += metric['score'] * weight
            total_weight += weight
            if metric['passed']:
                passed_metrics += 1
        
        # Calculate overall score
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Determine if enough metrics passed (6 out of 9)
        passed_overall = passed_metrics >= 6
        
        # Add overall results if not present
        if 'overall_results' not in result:
            result['overall_results'] = {
                'score': float(overall_score),
                'passed_metrics': passed_metrics,
                'total_metrics': total_metrics,
                'passed': passed_overall
            }
    
    # Generate summary if not present
    if 'summary' not in report:
        overall_scores = [r['overall_results']['score'] for r in report['results']]
        passed_qa_pairs = sum(1 for r in report['results'] if r['overall_results']['passed'])
        total_qa_pairs = len(report['results'])
        
        report['summary'] = {
            'total_qa_pairs': total_qa_pairs,
            'passed_qa_pairs': passed_qa_pairs,
            'pass_rate': (passed_qa_pairs / total_qa_pairs * 100) if total_qa_pairs > 0 else 0,
            'average_score': sum(overall_scores) / len(overall_scores) if overall_scores else 0
        }
    
    # Save updated report
    output_path = Path(report_path)
    backup_path = output_path.parent / f"{output_path.stem}_backup{output_path.suffix}"
    
    # Create backup of original file
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Save updated report
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Updated report saved to: {report_path}")
    print(f"Backup saved to: {backup_path}")

# Example usage:
if __name__ == "__main__":
    report_path = "evaluation_results/qwen_cpvs_no_context_agent/evaluation_report.json"
    update_evaluation_report(report_path)