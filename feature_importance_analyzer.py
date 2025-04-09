import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm
import seaborn as sns
from sklearn.model_selection import train_test_split

class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance for the PFIRec model
    """
    
    def __init__(self):
        """Initialize the analyzer"""
        pass
    
    def train_model_for_importance(self, dataset, idname, xnames):
        """
        Train a model to analyze feature importance
        
        Args:
            dataset (pd.DataFrame): Dataset with features and target
            idname (str): Group ID column name
            xnames (list): Feature names
            
        Returns:
            lightgbm.LGBMRanker: Trained model
        """
        # Split data for training and validation
        train_set, valid_set = train_test_split(dataset, test_size=0.2, random_state=42)
        
        # Prepare data for LGBMRanker
        qids_train = train_set.groupby(idname)[idname].count().to_numpy()
        X_train = train_set[xnames]
        y_train = train_set[["match"]]
        
        qids_validation = valid_set.groupby(idname)[idname].count().to_numpy()
        X_validation = valid_set[xnames]
        y_validation = valid_set[["match"]]
        
        # Train model
        model = lightgbm.LGBMRanker(
            objective="lambdarank",
            importance_type="gain"  # Use gain for feature importance
        )
        
        model.fit(
            X=X_train,
            y=y_train,
            group=qids_train,
            eval_set=[(X_validation, y_validation)],
            eval_group=[qids_validation],
            eval_at=5
        )
        
        return model
    
    def get_feature_importance(self, model, feature_names):
        """
        Get feature importance from a trained model
        
        Args:
            model: Trained model
            feature_names (list): Feature names
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        # Get feature importance
        importance = model.feature_importances_
        
        # Create dataframe
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        return feature_importance
    
    def plot_feature_importance(self, feature_importance, top_n=30, figsize=(12, 10)):
        """
        Plot feature importance
        
        Args:
            feature_importance (pd.DataFrame): Feature importance dataframe
            top_n (int): Number of top features to plot
            figsize (tuple): Figure size
        """
        # Get top N features
        top_features = feature_importance.head(top_n)
        
        # Plot
        plt.figure(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        print(f"Feature importance plot saved to 'feature_importance.png'")
    
    def get_feature_groups_importance(self, feature_importance):
        """
        Calculate importance by feature groups
        
        Args:
            feature_importance (pd.DataFrame): Feature importance dataframe
            
        Returns:
            pd.DataFrame: Feature group importance
        """
        # Define feature groups
        feature_groups = {
            'General OSS Experience': ["clsallcmt", "clsallpr", "clsalliss", "clspronum", "clsiss", 'clsallprreview'],
            'Activeness': ['clsonemonth_cmt', 'clstwomonth_cmt', 'clsthreemonth_cmt', 
                          'clsonemonth_pr', 'clstwomonth_pr', 'clsthreemonth_pr', 
                          'clsonemonth_iss', 'clstwomonth_iss', 'clsthreemonth_iss'],
            'Sentiment': ['clsissuesenmean', 'clsissuesenmedian', 'clsprsenmean', 'clsprsenmedian'],
            'Content Similarity': ['cmtjaccard_sim', 'cmtjaccard_sim_mean', "cmtcos_sim", "cmtcos_mean",
                                  "prjaccard_sim", 'prjaccard_sim_mean', "prcos_sim", "prcos_mean",
                                  'issjaccard_sim', 'issjaccard_sim_mean', "isscos_sim", "isscos_mean"],
            'Domain Similarity': ['readmecos_sim_mean', 'readmecos_sim', 'readmejaccard_sim_mean', 'readmejaccard_sim',
                                 "procos_mean", "procos_sim", "projaccard_mean", 'projaccard_sim'],
            'Issue Content': ['LengthOfTitle', 'LengthOfDescription', 'NumOfCode', 'NumOfUrls', 'NumOfPics',
                             'issuesen', 'coleman_liau_index', 'flesch_reading_ease', 'flesch_kincaid_grade', 
                             'automated_readability_index'],
            'Issue Labels': ['buglabelnum', 'featurelabelnum', 'testlabelnum', 'buildlabelnum', 'doclabelnum', 
                            'codinglabelnum', 'enhancelabelnum', 'gfilabelnum', 'mediumlabelnum', 'majorlabelnum', 
                            'triagedlabelnum', 'untriagedlabelnum', 'labelnum'],
            'Repository Background': ['pro_gfi_ratio', 'pro_gfi_num', 'proclspr', 'crtclsissnum', 'pro_star', 
                                     'openiss', 'openissratio', 'contributornum', 'procmt']
        }
        
        # Calculate group importance
        group_importance = []
        for group, features in feature_groups.items():
            # Filter features that exist in the importance dataframe
            existing_features = [f for f in features if f in feature_importance['Feature'].values]
            if existing_features:
                # Sum importance of features in the group
                group_imp = feature_importance[feature_importance['Feature'].isin(existing_features)]['Importance'].sum()
                group_importance.append({
                    'Group': group,
                    'Importance': group_imp,
                    'Feature Count': len(existing_features)
                })
        
        # Create dataframe
        group_importance_df = pd.DataFrame(group_importance)
        
        # Sort by importance
        group_importance_df = group_importance_df.sort_values('Importance', ascending=False)
        
        return group_importance_df
    
    def plot_group_importance(self, group_importance, figsize=(10, 6)):
        """
        Plot feature group importance
        
        Args:
            group_importance (pd.DataFrame): Feature group importance dataframe
            figsize (tuple): Figure size
        """
        # Plot
        plt.figure(figsize=figsize)
        sns.barplot(x='Importance', y='Group', data=group_importance)
        plt.title('Feature Group Importance')
        plt.tight_layout()
        plt.savefig('feature_group_importance.png')
        plt.close()
        
        print(f"Feature group importance plot saved to 'feature_group_importance.png'")
    
    def recommend_feature_reduction(self, feature_importance, threshold=0.95):
        """
        Recommend features to keep based on cumulative importance
        
        Args:
            feature_importance (pd.DataFrame): Feature importance dataframe
            threshold (float): Cumulative importance threshold
            
        Returns:
            list: Recommended features to keep
        """
        # Calculate cumulative importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        feature_importance['Cumulative Importance'] = feature_importance['Importance'].cumsum() / feature_importance['Importance'].sum()
        
        # Get features to keep
        features_to_keep = feature_importance[feature_importance['Cumulative Importance'] <= threshold]['Feature'].tolist()
        
        return features_to_keep
    
    def analyze_dataset(self, dataset, idname, xnames, output_prefix='feature_analysis'):
        """
        Analyze feature importance for a dataset
        
        Args:
            dataset (pd.DataFrame): Dataset with features and target
            idname (str): Group ID column name
            xnames (list): Feature names
            output_prefix (str): Prefix for output files
            
        Returns:
            tuple: (feature_importance, group_importance, features_to_keep)
        """
        print(f"Training model for feature importance analysis...")
        model = self.train_model_for_importance(dataset, idname, xnames)
        
        print(f"Calculating feature importance...")
        feature_importance = self.get_feature_importance(model, xnames)
        
        print(f"Top 10 most important features:")
        print(feature_importance.head(10))
        
        print(f"Plotting feature importance...")
        self.plot_feature_importance(feature_importance, top_n=30, figsize=(12, 10))
        
        print(f"Calculating feature group importance...")
        group_importance = self.get_feature_groups_importance(feature_importance)
        
        print(f"Feature group importance:")
        print(group_importance)
        
        print(f"Plotting feature group importance...")
        self.plot_group_importance(group_importance, figsize=(10, 6))
        
        print(f"Recommending feature reduction...")
        features_to_keep = self.recommend_feature_reduction(feature_importance, threshold=0.95)
        
        print(f"Recommended to keep {len(features_to_keep)} out of {len(xnames)} features ({len(features_to_keep)/len(xnames)*100:.1f}%)")
        
        # Save results
        feature_importance.to_csv(f'{output_prefix}_feature_importance.csv', index=False)
        group_importance.to_csv(f'{output_prefix}_group_importance.csv', index=False)
        pd.DataFrame({'Feature': features_to_keep}).to_csv(f'{output_prefix}_features_to_keep.csv', index=False)
        
        return feature_importance, group_importance, features_to_keep