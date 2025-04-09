import os
import sys
import pandas as pd
import argparse
from optimized_github_feature_collector import OptimizedGitHubFeatureCollector

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Collect optimized GitHub features in batch')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with user-issue pairs')
    parser.add_argument('--output', type=str, required=True, help='Output pickle file for features')
    parser.add_argument('--token', type=str, help='GitHub API token (or set GITHUB_TOKEN env var)')
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Check if token is provided
    token = args.token or os.environ.get('GITHUB_TOKEN')
    if not token:
        print("Please provide a GitHub API token via --token or set the GITHUB_TOKEN environment variable")
        sys.exit(1)
    
    # Load input data
    try:
        input_data = pd.read_csv(args.input)
        print(f"Loaded {len(input_data)} user-issue pairs from {args.input}")
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)
    
    # Check required columns
    required_columns = ['username', 'owner', 'repo', 'issue_number']
    missing_columns = [col for col in required_columns if col not in input_data.columns]
    if missing_columns:
        print(f"Input file is missing required columns: {', '.join(missing_columns)}")
        print(f"Required columns: {', '.join(required_columns)}")
        sys.exit(1)
    
    # Create collector
    collector = OptimizedGitHubFeatureCollector(token)
    
    # Process each user-issue pair
    all_features = []
    for i, row in input_data.iterrows():
        print(f"Processing {i+1}/{len(input_data)}: {row['username']} on {row['owner']}/{row['repo']}#{row['issue_number']}")
        
        try:
            features = collector.collect_optimized_features(
                row['username'], 
                row['owner'], 
                row['repo'], 
                row['issue_number']
            )
            
            if features:
                all_features.append(features)
                print(f"  Collected {len(features)} features")
            else:
                print(f"  No features collected")
        except Exception as e:
            print(f"  Error collecting features: {e}")
    
    # Create DataFrame and save
    if all_features:
        features_df = pd.DataFrame(all_features)
        collector.save_features(features_df, args.output)
        print(f"Saved {len(features_df)} feature records to {args.output}")
    else:
        print("No features collected")
    
    print("Done!")

if __name__ == "__main__":
    main()