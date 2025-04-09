import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from feature_importance_analyzer import FeatureImportanceAnalyzer

def main():
    """
    Main function to run feature importance analysis
    """
    print("Starting feature importance analysis...")
    
    # Create output directory if it doesn't exist
    os.makedirs("feature_analysis", exist_ok=True)
    
    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer()
    
    # Define feature groups as in model.py
    xnames_sub_cumu = [
        # General OSS experience
        "clsallcmt", "clsallpr", "clsalliss", "clspronum", "clsiss", 'clsallprreview',
    ]
    
    xnames_sub_act = [
        # Activeness
        'clsonemonth_cmt', 'clstwomonth_cmt', 'clsthreemonth_cmt', 
        'clsonemonth_pr', 'clstwomonth_pr', 'clsthreemonth_pr', 
        'clsonemonth_iss', 'clstwomonth_iss', 'clsthreemonth_iss',
    ]

    xnames_sub_sen = [
        # Sentiment
        'clsissuesenmean', 'clsissuesenmedian', 'clsprsenmean', 'clsprsenmedian',
    ]
    
    xnames_sub_clssolvediss = [
        # Expertise preference - Content preference
        'solvedisscos_sim', 'solvedisscos_mean',
        "solvedissjaccard_sim", "solvedissjaccard_sim_mean",
        "solvedissuelabel_sum", "solvedissuelabel_ratio",
    ]
    
    xnames_sub_clsrptiss = [
        # Expertise preference - Content preference
        'issjaccard_sim', 'issjaccard_sim_mean',
        "isscos_sim", "isscos_mean",
        'issuelabel_sum', 'issuelabel_ratio',
    ]
    
    xnames_sub_clscomtiss = [
        # Expertise preference - Content preference
        'commentissuelabel_sum', 'commentissuelabel_ratio',
        'commentissuecos_sim', 'commentissuecos_sim_mean',
        'commentissuejaccard_sim', 'commentissuejaccard_sim_mean',
    ]
    
    xnames_sub_clscmt = [
        # Expertise preference - Content preference
        'cmtjaccard_sim', 'cmtjaccard_sim_mean',
        "cmtcos_sim", "cmtcos_mean",
    ]
    
    xnames_sub_clspr = [
        # Expertise preference - Content preference
        "prjaccard_sim", 'prjaccard_sim_mean',
        "prcos_sim", "prcos_mean",
        'prlabel_sum', 'prlabel_ratio',
    ]
    
    xnames_sub_clsprreview = [
        # Expertise preference - Content preference
        'prreviewcos_sim', 'prreviewcos_sim_mean', 
        'prreviewjaccard_sim', 'prreviewjaccard_sim_mean',
    ]

    xnames_sub_clscont = xnames_sub_clscmt + xnames_sub_clspr + xnames_sub_clsprreview + \
                         xnames_sub_clsrptiss + xnames_sub_clscomtiss + xnames_sub_clssolvediss + ['lan_sim']
    
    xnames_sub_domain = [
        # Expertise preference - Domain preference
        'readmecos_sim_mean', 'readmecos_sim',
        'readmejaccard_sim_mean', 'readmejaccard_sim',
        "procos_mean", "procos_sim",
        "projaccard_mean", 'projaccard_sim',
        'prostopic_sum', 'prostopic_ratio',
    ]
    
    xnames_sub_isscont = [
        # Candidate issues - Content of issues
        'LengthOfTitle', 'LengthOfDescription', 'NumOfCode', 'NumOfUrls', 'NumOfPics', 
        'buglabelnum', 'featurelabelnum', 'testlabelnum', 'buildlabelnum', 'doclabelnum', 
        'codinglabelnum', 'enhancelabelnum', 'gfilabelnum', 'mediumlabelnum', 'majorlabelnum', 
        'triagedlabelnum', 'untriagedlabelnum', 'labelnum',
        'issuesen', 'coleman_liau_index', 'flesch_reading_ease', 'flesch_kincaid_grade', 'automated_readability_index',
    ]

    xnames_sub_back = [
        # Candidate issues - Background of issues
        'pro_gfi_ratio', 'pro_gfi_num', 'proclspr', 'crtclsissnum', 'pro_star', 'openiss', 
        'openissratio', 'contributornum', 'procmt',
        'rptcmt', 'rptiss', 'rptpr', 'rptpronum', 'rptallcmt', 'rptalliss', 'rptallpr', 
        'rpt_reviews_num_all', 'rpt_max_stars_commit', 'rpt_max_stars_issue', 'rpt_max_stars_pull', 
        'rpt_max_stars_review', 'rptisnew', 'rpt_gfi_ratio',
        'ownercmt', 'owneriss', 'ownerpr', 'ownerpronum', 'ownerallcmt', 'owneralliss', 'ownerallpr', 
        'owner_reviews_num_all', 'owner_max_stars_commit', 'owner_max_stars_issue', 'owner_max_stars_pull', 
        'owner_max_stars_review', 'owner_gfi_ratio', 'owner_gfi_num',
    ]

    # Combine all features
    xnames_LambdaMART = xnames_sub_cumu + xnames_sub_act + xnames_sub_sen + xnames_sub_clscont + \
                        xnames_sub_domain + xnames_sub_isscont + xnames_sub_back
    
    # Define dataset paths
    datasetname = 'simcse'  # Using SimCSE as default (PFIRec)
    path_name = "./data/dataset_"
    
    # Load and combine datasets
    print(f"Loading datasets for {datasetname}...")
    datasets = []
    for i in range(10):  # Load first 10 folds for analysis
        try:
            dataset = pd.read_pickle(f"{path_name}{datasetname}_{i}.pkl")
            datasets.append(dataset)
            print(f"Loaded dataset fold {i}")
        except Exception as e:
            print(f"Error loading dataset fold {i}: {e}")
    
    if not datasets:
        print("No datasets could be loaded. Exiting.")
        return
    
    # Combine datasets
    combined_dataset = pd.concat(datasets, axis=0)
    print(f"Combined dataset shape: {combined_dataset.shape}")
    
    # Define group ID column
    idname = "issgroupid"
    
    # Run analysis for all features
    print("\n=== Analyzing All Features ===")
    all_features_output = os.path.join("feature_analysis", "all_features")
    feature_importance, group_importance, features_to_keep = analyzer.analyze_dataset(
        combined_dataset, 
        idname=idname, 
        xnames=xnames_LambdaMART,
        output_prefix=all_features_output
    )
    
    # Run analysis for each feature group separately
    feature_groups = {
        "General_OSS_Experience": xnames_sub_cumu,
        "Activeness": xnames_sub_act,
        "Sentiment": xnames_sub_sen,
        "Content_Similarity": xnames_sub_clscont,
        "Domain_Similarity": xnames_sub_domain,
        "Issue_Content": xnames_sub_isscont,
        "Repository_Background": xnames_sub_back
    }
    
    for group_name, features in feature_groups.items():
        print(f"\n=== Analyzing {group_name} Features ===")
        group_output = os.path.join("feature_analysis", group_name)
        analyzer.analyze_dataset(
            combined_dataset, 
            idname=idname, 
            xnames=features,
            output_prefix=group_output
        )
    
    # Run analysis for different feature combinations
    feature_combinations = {
        "No_Similarity": xnames_sub_cumu + xnames_sub_act + xnames_sub_sen + xnames_sub_isscont + xnames_sub_back,
        "No_Developer": xnames_sub_clscont + xnames_sub_domain + xnames_sub_isscont + xnames_sub_back,
        "No_Issue": xnames_sub_cumu + xnames_sub_act + xnames_sub_sen + xnames_sub_clscont + xnames_sub_domain
    }
    
    for combo_name, features in feature_combinations.items():
        print(f"\n=== Analyzing {combo_name} Feature Combination ===")
        combo_output = os.path.join("feature_analysis", combo_name)
        analyzer.analyze_dataset(
            combined_dataset, 
            idname=idname, 
            xnames=features,
            output_prefix=combo_output
        )
    
    # Generate summary report
    print("\n=== Generating Summary Report ===")
    with open(os.path.join("feature_analysis", "summary_report.txt"), "w") as f:
        f.write("# Feature Importance Analysis Summary\n\n")
        
        f.write("## Top 20 Most Important Features\n\n")
        for i, row in feature_importance.head(20).iterrows():
            f.write(f"{i+1}. {row['Feature']}: {row['Importance']:.6f}\n")
        
        f.write("\n## Feature Group Importance\n\n")
        for i, row in group_importance.iterrows():
            f.write(f"{i+1}. {row['Group']}: {row['Importance']:.6f} ({row['Feature Count']} features)\n")
        
        f.write(f"\n## Feature Reduction\n\n")
        f.write(f"Original feature count: {len(xnames_LambdaMART)}\n")
        f.write(f"Recommended feature count: {len(features_to_keep)} ({len(features_to_keep)/len(xnames_LambdaMART)*100:.1f}%)\n\n")
        f.write("Recommended features to keep:\n")
        for feature in features_to_keep:
            f.write(f"- {feature}\n")
    
    print(f"Analysis complete! Results saved to the 'feature_analysis' directory.")
    print(f"Summary report: feature_analysis/summary_report.txt")

if __name__ == "__main__":
    main()