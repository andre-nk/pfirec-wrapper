import pandas as pd
import numpy as np
import os
import pickle
from collections import defaultdict

def analyze_dataset_structure():
    """Analyze the structure of the dataset to understand its components"""
    # Load the dataset
    datasetname = 'simcse'
    dataset_index = 2
    path_name = "./data/dataset_"
    dataset_path = f"{path_name}{datasetname}_{dataset_index}.pkl"
    
    print(f"Loading dataset from {dataset_path}")
    dataset = pd.read_pickle(dataset_path)
    
    # Basic dataset info
    print(f"\nDataset shape: {dataset.shape}")
    print(f"Number of columns: {len(dataset.columns)}")
    
    # Examine the structure
    idname = "issgroupid"
    group_ids = dataset[idname].unique()
    print(f"Number of unique groups: {len(group_ids)}")
    
    # Categorize columns
    column_categories = {
        "User features": [],
        "Issue features": [],
        "Repository features": [],
        "Similarity features": [],
        "Other features": []
    }
    
    # Define patterns for categorization
    user_patterns = ["cls", "user"]
    issue_patterns = ["Length", "NumOf", "labelnum", "issuesen"]
    repo_patterns = ["pro_", "openiss", "contributor"]
    similarity_patterns = ["_sim", "jaccard", "cos_"]
    
    # Categorize columns
    for col in dataset.columns:
        if any(pattern in col for pattern in user_patterns):
            column_categories["User features"].append(col)
        elif any(pattern in col for pattern in issue_patterns):
            column_categories["Issue features"].append(col)
        elif any(pattern in col for pattern in repo_patterns):
            column_categories["Repository features"].append(col)
        elif any(pattern in col for pattern in similarity_patterns):
            column_categories["Similarity features"].append(col)
        else:
            column_categories["Other features"].append(col)
    
    # Print column categories
    for category, columns in column_categories.items():
        print(f"\n{category} ({len(columns)}):")
        print(", ".join(columns[:10]) + ("..." if len(columns) > 10 else ""))
    
    # Analyze a sample group
    sample_group_id = group_ids[0]
    sample_group = dataset[dataset[idname] == sample_group_id]
    
    print(f"\nSample group: {sample_group_id}")
    print(f"Number of issues in group: {len(sample_group)}")
    print(f"Number of positive matches: {sample_group['match'].sum()}")
    
    # Check if user features are consistent within a group
    user_consistent = True
    for col in column_categories["User features"]:
        if col in dataset.columns and len(sample_group[col].unique()) > 1:
            user_consistent = False
            break
    
    print(f"User features consistent within group: {user_consistent}")
    
    # Check if repository features are consistent for each repository
    repo_features = defaultdict(list)
    for _, row in sample_group.iterrows():
        repo_key = f"{row.get('owner', '')}/{row.get('name', '')}"
        for col in column_categories["Repository features"]:
            if col in dataset.columns:
                repo_features[(repo_key, col)].append(row[col])
    
    repo_consistent = True
    for (repo_key, col), values in repo_features.items():
        if len(set(values)) > 1:
            repo_consistent = False
            break
    
    print(f"Repository features consistent for each repo: {repo_consistent}")
    
    # Analyze feature importance for matching
    print("\nFeature correlation with matching:")
    correlations = []
    for col in dataset.columns:
        if col != 'match' and col != idname:
            try:
                corr = dataset[col].corr(dataset['match'])
                if not np.isnan(corr):
                    correlations.append((col, corr))
            except:
                pass
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Print top correlations
    for col, corr in correlations[:10]:
        print(f"{col}: {corr:.4f}")
    
    # Identify essential features
    print("\nEssential features for a simplified model:")
    essential_features = []
    
    # Add top correlated features
    for col, corr in correlations[:15]:
        if abs(corr) > 0.05:  # Threshold for correlation
            essential_features.append(col)
    
    # Ensure we have some features from each category
    for category, columns in column_categories.items():
        if category != "Other features":
            # Add top 2 features from each category if not already included
            for col in columns[:2]:
                if col not in essential_features and col != 'match' and col != idname:
                    essential_features.append(col)
    
    print(", ".join(essential_features))
    
    return dataset, essential_features

if __name__ == "__main__":
    analyze_dataset_structure()