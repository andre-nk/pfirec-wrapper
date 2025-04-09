import os
import sys
import pandas as pd
from optimized_github_feature_collector import OptimizedGitHubFeatureCollector
import joblib
import re
import nltk


def ensure_nltk_resources():
    """
    Ensure all required NLTK resources are downloaded
    """
    # First, create NLTK data directory if it doesn't exist
    nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)

    # Add to NLTK data path
    nltk.data.path.insert(0, nltk_data_dir)

    # Core resources we need - only punkt is critical
    nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)

    # Create directory structure for punkt_tab if needed
    punkt_tab_dir = os.path.join(nltk_data_dir, "tokenizers", "punkt_tab", "english")
    os.makedirs(punkt_tab_dir, exist_ok=True)

    # Create an empty file in punkt_tab to satisfy the lookup
    import pathlib

    pathlib.Path(os.path.join(punkt_tab_dir, "punkt.pickle")).touch(exist_ok=True)


class IssueRecommender:
    """
    Recommends GitHub issues to a user based on their profile and repository
    """

    def __init__(self, model_path, token=None):
        """
        Initialize the recommender

        Args:
            model_path (str): Path to the trained model file
            token (str): GitHub API token (optional if GITHUB_TOKEN env var is set)
        """
        # Ensure NLTK resources are available
        ensure_nltk_resources()

        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError(
                "GitHub API token is required. Set GITHUB_TOKEN environment variable or pass token parameter."
            )

        self.collector = OptimizedGitHubFeatureCollector(self.token)

        # Load the trained model
        self.model = joblib.load(model_path)

    def parse_repo_url(self, repo_url):
        """
        Parse repository URL to extract owner and repo name

        Args:
            repo_url (str): GitHub repository URL

        Returns:
            tuple: (owner, repo_name)
        """
        # Match patterns like https://github.com/owner/repo or github.com/owner/repo
        match = re.match(r"(?:https?://)?github\.com/([^/]+)/([^/]+)", repo_url)
        if not match:
            raise ValueError(f"Invalid GitHub repository URL: {repo_url}")

        owner = match.group(1)
        repo_name = match.group(2)

        # Remove .git suffix if present
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        return owner, repo_name

    def ensure_all_features_present(self, features_df):
        """
        Ensure all features needed by the model are present in the DataFrame

        Args:
            features_df (pd.DataFrame): DataFrame of collected features

        Returns:
            pd.DataFrame: DataFrame with all required features
        """
        # Get feature columns used by the model
        model_features = self.model.feature_name_

        # Find missing features
        missing_features = set(model_features) - set(features_df.columns)

        # Add missing features with default value 0
        for feature in missing_features:
            features_df[feature] = 0

        print(f"Added {len(missing_features)} missing features with default value 0")

        # Ensure we only return features needed by the model
        return features_df[model_features]

    def recommend_issues(self, username, repo_url, limit=10, max_issues_to_process=50):
        """
        Recommend issues from a repository to a user

        Args:
            username (str): GitHub username
            repo_url (str): GitHub repository URL
            limit (int): Maximum number of issues to recommend
            max_issues_to_process (int): Maximum number of issues to analyze

        Returns:
            list: Recommended issues with scores
        """
        # Parse repository URL
        owner, repo_name = self.parse_repo_url(repo_url)
        print(f"Finding issues for {username} in {owner}/{repo_name}")

        # Get repository information
        repo_info = self.collector.get_repo_info(owner, repo_name)
        if not repo_info:
            raise ValueError(f"Repository not found: {owner}/{repo_name}")

        # Get open issues
        issues = self.collector.get_repo_issues(owner, repo_name, state="open")
        if not issues:
            print(f"No open issues found in {owner}/{repo_name}")
            return []

        print(f"Found {len(issues)} open issues. Analyzing compatibility...")

        # Pre-fetch user data once (this is the key optimization)
        user_data = self.collector.get_user_info(username)
        user_repos = self.collector.get_user_repos(username)

        # Batch fetch user commits and PRs in a single operation where possible
        try:
            user_commits = self.collector.get_user_commits(username)
        except Exception as e:
            print(f"Error fetching user commits: {e}")
            user_commits = []

        try:
            user_prs = self.collector.get_user_prs(username)
        except Exception as e:
            print(f"Error fetching user PRs: {e}")
            user_prs = []

        # Pre-fetch repository data once
        try:
            repo_readme = self.collector.get_repo_readme(owner, repo_name)
        except Exception as e:
            print(f"Error fetching repo README: {e}")
            repo_readme = ""

        try:
            repo_contributors = self.collector.get_repo_contributors(owner, repo_name)
        except Exception as e:
            print(f"Error fetching repo contributors: {e}")
            repo_contributors = []

        # Limit the number of issues to process
        issues_to_process = issues[: min(max_issues_to_process, len(issues))]

        # Prepare user data for similarity calculation once
        user_data_for_sim = {
            "issues": user_prs,  # Reuse PRs as issues
            "prs": user_prs,
            "commits": user_commits,
        }

        # Prepare repo data for similarity calculation once
        repo_data_for_sim = {"readme": repo_readme}

        # Set up progress tracking
        num_issues = len(issues_to_process)
        print(f"Processing {num_issues} issues...")

        # Collect features for each issue
        issue_features = []
        for i, issue in enumerate(issues_to_process):
            issue_number = issue["number"]
            print(f"Processing issue #{issue_number} ({i+1}/{num_issues})")

            try:
                # Pass pre-fetched data to avoid redundant API calls
                features = (
                    self.collector.collect_optimized_features_with_prefetched_data(
                        username,
                        owner,
                        repo_name,
                        issue_number,
                        user_data=user_data,
                        user_repos=user_repos,
                        user_commits=user_commits,
                        user_prs=user_prs,
                        repo_readme=repo_readme,
                        repo_contributors=repo_contributors,
                        issue_data=issue,  # Pass the issue data we already have
                    )
                )

                if features:
                    # Add issue title and URL for reference
                    features["issue_title"] = issue["title"]
                    features["issue_url"] = issue["html_url"]
                    issue_features.append(features)
            except Exception as e:
                print(f"Error processing issue #{issue_number}: {e}")

        if not issue_features:
            print("No features could be collected for any issues")
            return []

        # Convert to DataFrame
        features_df = pd.DataFrame(issue_features)

        # Handle missing features needed by the model
        features_for_model = self.ensure_all_features_present(features_df)

        # Make predictions
        scores = self.model.predict(features_for_model)

        # Add scores to features
        features_df["match_score"] = scores

        # Sort by score (descending)
        sorted_issues = features_df.sort_values("match_score", ascending=False)

        # Return top recommendations
        recommendations = []
        for _, row in sorted_issues.head(limit).iterrows():
            recommendations.append(
                {
                    "issue_number": int(row["issue_number"]),
                    "issue_title": row["issue_title"],
                    "issue_url": row["issue_url"],
                    "match_score": float(row["match_score"]),
                }
            )

        return recommendations


if __name__ == "__main__":
    # Check if token is provided
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Please set the GITHUB_TOKEN environment variable")
        sys.exit(1)

    # Path to trained model
    model_path = "./saved_models/pfirec_model.pkl"

    # Create recommender
    recommender = IssueRecommender(model_path, token)

    # Example: Recommend issues for a user
    username = "andre-nk"  # Replace with actual GitHub username
    repo_url = "https://github.com/openfoodfacts/smooth-app"  # Replace with actual repository URL

    try:
        recommendations = recommender.recommend_issues(username, repo_url, limit=10)

        print("\nRecommended Issues:")
        for i, rec in enumerate(recommendations):
            print(f"{i+1}. {rec['issue_title']} (#{rec['issue_number']})")
            print(f"   Score: {rec['match_score']:.4f}")
            print(f"   URL: {rec['issue_url']}")
            print()
    except Exception as e:
        print(f"Error: {e}")
