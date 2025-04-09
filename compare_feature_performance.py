import pandas as pd
import numpy as np
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import argparse
import os

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Compare model performance with all vs optimized features')
    parser.add_argument('--full-features', type=str, required=True, help='Pickle file with all features')
    parser.add_argument('--optimized-features', type=str, required=True, help='Pickle file with optimized features')
    parser.add_argument('--output-dir', type=str, default='performance_comparison', help='Output directory for results')
    return parser.parse_args()

def train_and_evaluate(X_train, y_train, X_test, y_test, qids_train, qids_test):
    """Train model and evaluate performance"""
    model = lightgbm.LGBMRanker(
        objective="lambdarank",
        boosting_type="gbdt",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(
        X=X_train,
        y=y_train,
        group=qids_train,
        eval_set=[(X_test, y_test)],
        eval_group=[qids_test],
        eval_at=5,
        verbose=False
    )
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate NDCG@k for different k values
    ndcg_scores = {}
    for k in [1, 3, 5, 10]:
        ndcg_scores[f'ndcg@{k}'] = ndcg_score(
            np.array([y_test]), 
            np.array([y_pred]), 
            k=k
        )
    
    return model, ndcg_scores

def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    print(f"Loading full features from {args.full_features}")
    full_df = pd.read_pickle(args.full_features)
    
    print(f"Loading optimized features from {args.optimized_features}")
    optimized_df = pd.read_pickle(args.optimized_features)
    
    # Get feature lists
    full_features = [col for col in full_df.columns if col not in ['username', 'owner', 'repo', 'issue_number', 'match', 'issgroupid']]
    optimized_features = [col for col in optimized_df.columns if col not in ['username', 'owner', 'repo', 'issue_number', 'match', 'issgroupid']]
    
    print(f"Full features: {len(full_features)}")
    print(f"Optimized features: {len(optimized_features)}")
    
    # Split data
    train_full, test_full = train_test_split(full_df, test_size=0.2, random_state=42)
    train_opt, test_opt = train_test_split(optimized_df, test_size=0.2, random_state=42)
    
    # Prepare data for full features
    X_train_full = train_full[full_features]
    y_train_full = train_full['match']
    qids_train_full = train_full.groupby('issgroupid')['issgroupid'].count().to_numpy()
    
    X_test_full = test_full[full_features]
    y_test_full = test_full['match']
    qids_test_full = test_full.groupby('issgroupid')['issgroupid'].count().to_numpy()
    
    # Prepare data for optimized features
    X_train_opt = train_opt[optimized_features]
    y_train_opt = train_opt['match']
    qids_train_opt = train_opt.groupby('issgroupid')['issgroupid'].count().to_numpy()
    
    X_test_opt = test_opt[optimized_features]
    y_test_opt = test_opt['match']
    qids_test_opt = test_opt.groupby('issgroupid')['issgroupid'].count().to_numpy()
    
    # Train and evaluate models
    print("Training model with full features...")
    model_full, ndcg_full = train_and_evaluate(
        X_train_full, y_train_full, X_test_full, y_test_full, qids_train_full, qids_test_full
    )
    
    print("Training model with optimized features...")
    model_opt, ndcg_opt = train_and_evaluate(
        X_train_opt, y_train_opt, X_test_opt, y_test_opt, qids_train_opt, qids_test_opt
    )
    
    # Compare results
    print("\nPerformance comparison:")
    for k in [1, 3, 5, 10]:
        metric = f'ndcg@{k}'
        full_score = ndcg_full[metric]
        opt_score = ndcg_opt[metric]
        diff = opt_score - full_score
        diff_pct = (diff / full_score) * 100
        
        print(f"{metric}: Full = {full_score:.4f}, Optimized = {opt_score:.4f}, Diff = {diff:.4f} ({diff_pct:+.2f}%)")
    
    # Plot comparison
    metrics = [f'ndcg@{k}' for k in [1, 3, 5, 10]]
    full_scores = [ndcg_full[m] for m in metrics]
    opt_scores = [ndcg_opt[m] for m in metrics]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, full_scores, width, label='Full Features')
    plt.bar(x + width/2, opt_scores, width, label='Optimized Features')
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Performance Comparison: Full vs Optimized Features')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(args.output_dir, 'performance_comparison.png'))
    print(f"Performance comparison plot saved to {args.output_dir}/performance_comparison.png")
    
    # Save feature importance
    feature_importance_full = pd.DataFrame({
        'Feature': full_features,
        'Importance': model_full.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    feature_importance_opt = pd.DataFrame({
        'Feature': optimized_features,
        'Importance': model_opt.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    feature_importance_full.to_csv(os.path.join(args.output_dir, 'feature_importance_full.csv'), index=False)
    feature_importance_opt.to_csv(os.path.join(args.output_dir, 'feature_importance_opt.csv'), index=False)
    
    print(f"Feature importance saved to {args.output_dir}/feature_importance_*.csv")
    
    # Save summary
    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write("# Performance Comparison: Full vs Optimized Features\n\n")
        
        f.write("## Feature Counts\n")
        f.write(f"Full features: {len(full_features)}\n")
        f.write(f"Optimized features: {len(optimized_features)}\n")
        f.write(f"Reduction: {len(full_features) - len(optimized_features)} features ({(1 - len(optimized_features)/len(full_features))*100:.1f}%)\n\n")
        
        f.write("## Performance Metrics\n")
        for k in [1, 3, 5, 10]:
            metric = f'ndcg@{k}'
            full_score = ndcg_full[metric]
            opt_score = ndcg_opt[metric]
            diff = opt_score - full_score
            diff_pct = (diff / full_score) * 100
            
            f.write(f"{metric}: Full = {full_score:.4f}, Optimized = {opt_score:.4f}, Diff = {diff:.4f} ({diff_pct:+.2f}%)\n")
    
    print(f"Summary saved to {args.output_dir}/summary.txt")
    print("Done!")

if __name__ == "__main__":
    main()