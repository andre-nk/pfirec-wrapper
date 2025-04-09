import os
import time
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import re
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import spacy
import sys

# Create a directory for NLTK data if it doesn't exist
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Set the NLTK data path
nltk.data.path.append(nltk_data_dir)


# Helper function to download NLTK resources
def download_nltk_resource(resource_name):
    try:
        nltk.data.find(resource_name)
        print(f"NLTK {resource_name} already downloaded")
        return True
    except LookupError:
        print(f"Downloading NLTK {resource_name}...")
        try:
            nltk.download(
                resource_name.split("/")[-1], download_dir=nltk_data_dir, quiet=False
            )
            print(f"Download of {resource_name} complete")
            return True
        except Exception as e:
            print(f"Error downloading {resource_name}: {e}")
            return False


# Download required NLTK resources
required_resources = ["tokenizers/punkt", "tokenizers/punkt_tab"]

for resource in required_resources:
    success = download_nltk_resource(resource)
    if not success:
        print(f"WARNING: Failed to download {resource}, using fallback methods")

# Verify that punkt is available
try:
    nltk.data.find("tokenizers/punkt")
    print("NLTK punkt tokenizer is now available")
    # Try a quick tokenization test
    test_text = "Hello world. This is a test."
    tokens = sent_tokenize(test_text)
    print(f"Tokenizer test successful: {tokens}")
except Exception as e:
    print(f"Tokenizer test failed: {e}")
    print("Attempting to fix by forcing download...")
    nltk.download("punkt", download_dir=nltk_data_dir)

# Also download punkt_tab directly to avoid the error
try:
    nltk.download("punkt", download_dir=nltk_data_dir)
    # Create directory structure for punkt_tab if needed
    punkt_tab_dir = os.path.join(nltk_data_dir, "tokenizers", "punkt_tab", "english")
    os.makedirs(punkt_tab_dir, exist_ok=True)
    print("Created punkt_tab directory structure")
except Exception as e:
    print(f"Error setting up punkt_tab: {e}")

# Load spaCy model for lemmatization
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If model not installed, download it
    import subprocess

    subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


class OptimizedGitHubFeatureCollector:
    """
    Optimized collector that only fetches the recommended features from feature importance analysis
    """

    def __init__(self, token=None):
        """
        Initialize the collector with GitHub API token

        Args:
            token (str): GitHub API token
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError(
                "GitHub API token is required. Set GITHUB_TOKEN environment variable or pass token parameter."
            )

        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }
        self.base_url = "https://api.github.com"
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = 0
        self.cache = {}

        # Add tokenization cache
        self.tokenization_cache = {}

    def _make_request(self, endpoint, params=None):
        """
        Make a request to GitHub API with rate limit handling

        Args:
            endpoint (str): API endpoint
            params (dict): Query parameters

        Returns:
            dict: Response JSON
        """
        # Check rate limit
        if self.rate_limit_remaining < 10:
            current_time = time.time()
            if current_time < self.rate_limit_reset:
                sleep_time = self.rate_limit_reset - current_time + 1
                print(
                    f"Rate limit almost reached. Sleeping for {sleep_time:.2f} seconds."
                )
                time.sleep(sleep_time)

        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)

        # Update rate limit info
        self.rate_limit_remaining = int(
            response.headers.get("X-RateLimit-Remaining", 0)
        )
        self.rate_limit_reset = int(response.headers.get("X-RateLimit-Reset", 0))

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            print(f"Resource not found: {url}")
            return None
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None

    def get_user_info(self, username):
        """
        Get basic user information

        Args:
            username (str): GitHub username

        Returns:
            dict: User information
        """
        cache_key = f"user_{username}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        user_data = self._make_request(f"users/{username}")
        if user_data:
            self.cache[cache_key] = user_data
        return user_data

    def get_user_repos(self, username):
        """
        Get repositories owned by a user

        Args:
            username (str): GitHub username

        Returns:
            list: User repositories
        """
        cache_key = f"user_repos_{username}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        repos = []
        page = 1
        while True:
            params = {"per_page": 100, "page": page}
            repo_data = self._make_request(f"users/{username}/repos", params)
            if not repo_data or len(repo_data) == 0:
                break
            repos.extend(repo_data)
            page += 1

        self.cache[cache_key] = repos
        return repos

    def get_repo_info(self, owner, repo):
        """
        Get repository information

        Args:
            owner (str): Repository owner
            repo (str): Repository name

        Returns:
            dict: Repository information
        """
        cache_key = f"repo_{owner}_{repo}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        repo_data = self._make_request(f"repos/{owner}/{repo}")
        if repo_data:
            self.cache[cache_key] = repo_data
        return repo_data

    def get_repo_issues(self, owner, repo, state="all"):
        """
        Get issues from a repository

        Args:
            owner (str): Repository owner
            repo (str): Repository name
            state (str): Issue state (open, closed, all)

        Returns:
            list: Repository issues
        """
        cache_key = f"repo_issues_{owner}_{repo}_{state}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        issues = []
        page = 1
        while True:
            params = {"per_page": 100, "page": page, "state": state}
            issue_data = self._make_request(f"repos/{owner}/{repo}/issues", params)
            if not issue_data or len(issue_data) == 0:
                break
            issues.extend(issue_data)
            page += 1

        self.cache[cache_key] = issues
        return issues

    def get_user_commits(self, username, since=None):
        """
        Get commits by a user

        Args:
            username (str): GitHub username
            since (datetime): Only commits after this date

        Returns:
            list: User commits
        """
        # This is a complex query that requires searching across repositories
        # For simplicity, we'll get commits from user's repositories
        commits = []
        repos = self.get_user_repos(username)

        for repo in repos:
            owner = repo["owner"]["login"]
            repo_name = repo["name"]

            params = {"author": username, "per_page": 100}
            if since:
                params["since"] = since.isoformat()

            repo_commits = self._make_request(
                f"repos/{owner}/{repo_name}/commits", params
            )
            if repo_commits:
                commits.extend(repo_commits)

        return commits

    def get_user_prs(self, username, state="all"):
        """
        Get pull requests by a user

        Args:
            username (str): GitHub username
            state (str): PR state (open, closed, all)

        Returns:
            list: User pull requests
        """
        # GitHub API doesn't have a direct endpoint for user PRs
        # We'll use the search API
        query = f"author:{username} type:pr state:{state}"
        params = {"q": query, "per_page": 100}

        prs = []
        page = 1
        while True:
            params["page"] = page
            search_results = self._make_request("search/issues", params)
            if not search_results or len(search_results.get("items", [])) == 0:
                break
            prs.extend(search_results["items"])
            page += 1

            # Search API has stricter rate limits
            time.sleep(1)

        return prs

    def get_repo_readme(self, owner, repo):
        """
        Get repository README content

        Args:
            owner (str): Repository owner
            repo (str): Repository name

        Returns:
            str: README content
        """
        cache_key = f"readme_{owner}_{repo}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        readme_data = self._make_request(f"repos/{owner}/{repo}/readme")
        if readme_data and "content" in readme_data:
            import base64

            content = base64.b64decode(readme_data["content"]).decode("utf-8")
            self.cache[cache_key] = content
            return content
        return ""

    def get_repo_contributors(self, owner, repo):
        """
        Get repository contributors

        Args:
            owner (str): Repository owner
            repo (str): Repository name

        Returns:
            list: Contributors
        """
        cache_key = f"contributors_{owner}_{repo}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        contributors = self._make_request(f"repos/{owner}/{repo}/contributors")
        if contributors:
            self.cache[cache_key] = contributors
        return contributors or []

    def lemmatize_text(self, text):
        """
        Lemmatize text using spaCy

        Args:
            text (str): Input text

        Returns:
            str: Lemmatized text
        """
        if not text:
            return ""

        try:
            doc = nlp(text)
            return " ".join([token.lemma_ for token in doc])
        except Exception as e:
            print(f"Error lemmatizing text: {e}")
            return text

    def calculate_cosine_similarity(self, emb0, emb1):
        """
        Calculate cosine similarity between two embeddings

        Args:
            emb0 (array-like): First embedding
            emb1 (array-like): Second embedding

        Returns:
            float: Cosine similarity score
        """
        try:
            return cosine_similarity(emb0, emb1)[0][0]
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0

    def calculate_jaccard_similarity(self, text1, text2):
        """
        Calculate Jaccard similarity between two texts

        Args:
            text1 (str): First text
            text2 (str): Second text

        Returns:
            float: Jaccard similarity score
        """
        if not text1 or not text2:
            return 0.0

        try:
            # Lemmatize texts
            text1 = self.lemmatize_text(text1)
            text2 = self.lemmatize_text(text2)

            # Create sets of words
            set1 = set(text1.split())
            set2 = set(text2.split())

            # Calculate intersection and union
            intersection = set1.intersection(set2)
            union = set1.union(set2)

            # Calculate Jaccard similarity
            return float(len(intersection)) / len(union) if union else 0.0
        except Exception as e:
            print(f"Error calculating Jaccard similarity: {e}")
            return 0.0

    def calculate_text_similarity(self, text1, text2, method="jaccard"):
        """
        Calculate similarity between two texts

        Args:
            text1 (str): First text
            text2 (str): Second text
            method (str): Similarity method (jaccard, cosine)

        Returns:
            float: Similarity score
        """
        if not text1 or not text2:
            return 0.0

        if method == "jaccard":
            return self.calculate_jaccard_similarity(text1, text2)

        # For cosine similarity
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            return self.calculate_cosine_similarity(
                tfidf_matrix[0:1], tfidf_matrix[1:2]
            )
        except Exception as e:
            print(f"Error in vectorization for {method} similarity: {e}")
            return 0.0

    def calculate_sentiment(self, text):
        """
        Calculate sentiment of text

        Args:
            text (str): Input text

        Returns:
            float: Sentiment score (-1 to 1)
        """
        if not text:
            return 0.0

        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0

    def calculate_readability(self, text):
        """
        Calculate readability metrics for text with robust error handling

        Args:
            text (str): Input text

        Returns:
            dict: Readability metrics
        """
        if not text:
            return {
                "coleman_liau_index": 0,
                "flesch_reading_ease": 0,
                "flesch_kincaid_grade": 0,
                "automated_readability_index": 0,
            }

        # Check if we already processed this text
        text_hash = hash(text)
        if text_hash in self.tokenization_cache:
            return self.tokenization_cache[text_hash]

        # Count characters, words, and sentences
        char_count = len(text)
        word_count = len(text.split())

        # Try multiple approaches to sentence tokenization with fallbacks
        try:
            # Try NLTK's tokenizer first - without verbose logging
            sentence_count = max(1, len(sent_tokenize(text)))
        except Exception as e:
            try:
                # Fall back to a simpler regex-based approach
                import re

                sentences = re.split(r"[.!?]+", text)
                sentence_count = max(1, len([s for s in sentences if s.strip()]))
            except Exception as e2:
                # Last resort: simple character counting
                sentence_endings = [".", "!", "?"]
                sentence_count = sum(1 for char in text if char in sentence_endings)
                sentence_count = max(1, sentence_count)  # Ensure at least 1 sentence

        # Average characters per word
        avg_chars_per_word = char_count / max(1, word_count)

        # Average words per sentence
        avg_words_per_sentence = word_count / max(1, sentence_count)

        # Calculate readability metrics
        coleman_liau = (
            0.0588 * (avg_chars_per_word * 100)
            - 0.296 * (100 / avg_words_per_sentence)
            - 15.8
        )

        flesch_reading_ease = (
            206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_chars_per_word)
        )

        flesch_kincaid_grade = (
            0.39 * avg_words_per_sentence + 11.8 * avg_chars_per_word - 15.59
        )

        automated_readability_index = (
            4.71 * avg_chars_per_word + 0.5 * avg_words_per_sentence - 21.43
        )

        # Store results in cache
        result = {
            "coleman_liau_index": max(0, coleman_liau),
            "flesch_reading_ease": max(0, flesch_reading_ease),
            "flesch_kincaid_grade": max(0, flesch_kincaid_grade),
            "automated_readability_index": max(0, automated_readability_index),
        }
        self.tokenization_cache[text_hash] = result

        return result

    def extract_optimized_issue_features(self, issue):
        """
        Extract only the important issue features based on feature importance analysis

        Args:
            issue (dict): Issue data

        Returns:
            dict: Important issue features
        """
        features = {}

        # Important issue content features
        features["LengthOfTitle"] = len(issue.get("title", ""))
        features["LengthOfDescription"] = (
            len(issue.get("body", "")) if issue.get("body") else 0
        )

        # Count code blocks and URLs in body
        body = issue.get("body", "") or ""
        features["NumOfCode"] = body.count("```")
        features["NumOfUrls"] = len(re.findall(r"https?://\S+", body))

        # Sentiment
        features["issuesen"] = self.calculate_sentiment(
            issue.get("title", "") + " " + (issue.get("body", "") or "")
        )

        # Readability
        readability = self.calculate_readability(
            issue.get("title", "") + " " + (issue.get("body", "") or "")
        )
        features["coleman_liau_index"] = readability["coleman_liau_index"]
        features["flesch_reading_ease"] = readability["flesch_reading_ease"]
        features["flesch_kincaid_grade"] = readability["flesch_kincaid_grade"]
        features["automated_readability_index"] = readability[
            "automated_readability_index"
        ]

        # Important label
        labels = [label["name"].lower() for label in issue.get("labels", [])]
        features["gfilabelnum"] = sum(
            1
            for label in labels
            if "good first issue" in label or "good-first-issue" in label
        )

        return features

    def extract_optimized_user_features(
        self,
        username,
        user_data=None,
        user_repos=None,
        user_commits=None,
        user_prs=None,
    ):
        """
        Extract only the important user features based on feature importance analysis

        Args:
            username (str): GitHub username
            user_data (dict, optional): Pre-fetched user information
            user_repos (list, optional): Pre-fetched user repositories
            user_commits (list, optional): Pre-fetched user commits
            user_prs (list, optional): Pre-fetched user pull requests

        Returns:
            dict: Important user features
        """
        features = {}

        # Use pre-fetched data or fetch if not provided
        user_info = user_data or self.get_user_info(username)
        if not user_info:
            return features

        user_repos = user_repos or self.get_user_repos(username)

        # Get user commits (last 3 months)
        three_months_ago = datetime.now() - timedelta(days=90)
        commits = user_commits or self.get_user_commits(
            username, since=three_months_ago
        )

        # Get user PRs
        prs = user_prs or self.get_user_prs(username)

        # Get user issues
        issues = []
        for repo in user_repos:
            repo_issues = self.get_repo_issues(repo["owner"]["login"], repo["name"])
            user_issues = [
                issue
                for issue in repo_issues
                if issue.get("user", {}).get("login") == username
            ]
            issues.extend(user_issues)

        # Important general OSS experience features
        features["clsallcmt"] = len(commits)
        features["clsiss"] = len(issues)
        features["clspronum"] = len(user_repos)

        # Important activeness features
        one_month_ago = datetime.now() - timedelta(days=30)
        commit_dates = [
            datetime.strptime(commit["commit"]["author"]["date"], "%Y-%m-%dT%H:%M:%SZ")
            for commit in commits
            if "commit" in commit
            and "author" in commit["commit"]
            and "date" in commit["commit"]["author"]
        ]
        features["clsonemonth_cmt"] = sum(
            1 for date in commit_dates if date >= one_month_ago
        )

        # Important sentiment features
        issue_sentiments = [
            self.calculate_sentiment(
                issue.get("title", "") + " " + (issue.get("body", "") or "")
            )
            for issue in issues
        ]
        pr_sentiments = [
            self.calculate_sentiment(
                pr.get("title", "") + " " + (pr.get("body", "") or "")
            )
            for pr in prs
        ]

        features["clsissuesenmean"] = (
            np.mean(issue_sentiments) if issue_sentiments else 0
        )
        features["clsissuesenmedian"] = (
            np.median(issue_sentiments) if issue_sentiments else 0
        )
        features["clsprsenmean"] = np.mean(pr_sentiments) if pr_sentiments else 0
        features["clsprsenmedian"] = np.median(pr_sentiments) if pr_sentiments else 0

        # Reporter features (simplified)
        features["rptcmt"] = len(commits)
        features["rptiss"] = len(issues)
        features["rptpr"] = len(prs)
        features["rptpronum"] = len(user_repos)
        features["rptallcmt"] = len(commits)
        features["rptalliss"] = len(issues)

        # Star metrics (simplified)
        features["rpt_max_stars_commit"] = (
            max([repo.get("stargazers_count", 0) for repo in user_repos])
            if user_repos
            else 0
        )
        features["rpt_max_stars_issue"] = features["rpt_max_stars_commit"]  # Simplified

        # GFI ratio
        gfi_count = 0
        for repo in user_repos:
            repo_issues = self.get_repo_issues(repo["owner"]["login"], repo["name"])
            gfi_issues = [
                issue
                for issue in repo_issues
                if any(
                    "good first issue" in label["name"].lower()
                    or "good-first-issue" in label["name"].lower()
                    for label in issue.get("labels", [])
                )
            ]
            gfi_count += len(gfi_issues)

        features["rpt_gfi_ratio"] = gfi_count / max(1, len(issues)) if issues else 0

        return features

    def extract_optimized_repo_features(
        self,
        owner,
        repo_name,
        repo_readme=None,
        repo_contributors=None,
        repo_info=None,
        issues=None,
    ):
        """
        Extract only the important repository features based on feature importance analysis

        Args:
            owner (str): Repository owner
            repo_name (str): Repository name
            repo_readme (str, optional): Pre-fetched repository README
            repo_contributors (list, optional): Pre-fetched repository contributors
            repo_info (dict, optional): Pre-fetched repository information
            issues (list, optional): Pre-fetched repository issues

        Returns:
            dict: Important repository features
        """
        features = {}

        # Get repository information
        repo_info = repo_info or self.get_repo_info(owner, repo_name)
        if not repo_info:
            return features

        # Important repository metrics
        features["pro_star"] = repo_info.get("stargazers_count", 0)
        features["procmt"] = repo_info.get(
            "size", 0
        )  # Size is a rough proxy for commits

        # Issues
        issues = issues or self.get_repo_issues(owner, repo_name)
        open_issues = [issue for issue in issues if issue.get("state") == "open"]
        features["openiss"] = len(open_issues)
        features["openissratio"] = (
            len(open_issues) / max(1, len(issues)) if issues else 0
        )

        # Good First Issues
        gfi_issues = [
            issue
            for issue in issues
            if any(
                "good first issue" in label["name"].lower()
                or "good-first-issue" in label["name"].lower()
                for label in issue.get("labels", [])
            )
        ]
        features["pro_gfi_num"] = len(gfi_issues)
        features["pro_gfi_ratio"] = (
            len(gfi_issues) / max(1, len(issues)) if issues else 0
        )

        # Contributors
        contributors = repo_contributors or self.get_repo_contributors(owner, repo_name)
        features["contributornum"] = len(contributors)

        # Critical issues (approximation)
        critical_issues = [
            issue
            for issue in issues
            if any(
                "critical" in label["name"].lower() or "high" in label["name"].lower()
                for label in issue.get("labels", [])
            )
        ]
        features["crtclsissnum"] = len(critical_issues)

        # Pull requests (approximation)
        features["proclspr"] = repo_info.get("open_issues_count", 0) - len(open_issues)

        return features

    def calculate_optimized_similarity_features(self, user_data, issue_data, repo_data):
        """
        Calculate only the important similarity features based on feature importance analysis

        Args:
            user_data (dict): User data
            issue_data (dict): Issue data
            repo_data (dict): Repository data

        Returns:
            dict: Important similarity features
        """
        features = {}

        # Get user issues, PRs, commits text
        user_issues_text = " ".join(
            [
                issue.get("title", "") + " " + (issue.get("body", "") or "")
                for issue in user_data.get("issues", [])
            ]
        )
        user_prs_text = " ".join(
            [
                pr.get("title", "") + " " + (pr.get("body", "") or "")
                for pr in user_data.get("prs", [])
            ]
        )
        user_commits_text = " ".join(
            [
                commit.get("commit", {}).get("message", "")
                for commit in user_data.get("commits", [])
            ]
        )

        # Get issue text
        issue_text = (
            issue_data.get("title", "") + " " + (issue_data.get("body", "") or "")
        )

        # Get repo text (README)
        repo_readme = repo_data.get("readme", "")

        # Important similarity features
        features["issjaccard_sim"] = self.calculate_text_similarity(
            user_issues_text, issue_text, "jaccard"
        )
        features["issjaccard_sim_mean"] = features["issjaccard_sim"]  # Simplified
        features["isscos_sim"] = self.calculate_text_similarity(
            user_issues_text, issue_text, "cosine"
        )
        features["isscos_mean"] = features["isscos_sim"]  # Simplified

        features["cmtcos_sim"] = self.calculate_text_similarity(
            user_commits_text, issue_text, "cosine"
        )
        features["cmtcos_mean"] = features["cmtcos_sim"]  # Simplified
        features["cmtjaccard_sim"] = self.calculate_text_similarity(
            user_commits_text, issue_text, "jaccard"
        )
        features["cmtjaccard_sim_mean"] = features["cmtjaccard_sim"]  # Simplified

        features["prcos_sim"] = self.calculate_text_similarity(
            user_prs_text, issue_text, "cosine"
        )
        features["prcos_mean"] = features["prcos_sim"]  # Simplified
        features["prjaccard_sim"] = self.calculate_text_similarity(
            user_prs_text, issue_text, "jaccard"
        )
        features["prjaccard_sim_mean"] = features["prjaccard_sim"]  # Simplified

        features["readmecos_sim"] = self.calculate_text_similarity(
            user_issues_text + user_prs_text + user_commits_text, repo_readme, "cosine"
        )
        features["readmecos_sim_mean"] = features["readmecos_sim"]  # Simplified

        features["procos_sim"] = self.calculate_text_similarity(
            user_issues_text + user_prs_text, repo_readme, "cosine"
        )
        features["procos_mean"] = features["procos_sim"]  # Simplified
        features["projaccard_mean"] = self.calculate_text_similarity(
            user_issues_text + user_prs_text, repo_readme, "jaccard"
        )

        # Comment issue similarity (simplified)
        features["commentissuecos_sim"] = self.calculate_text_similarity(
            user_commits_text, issue_text, "cosine"
        )
        features["commentissuecos_sim_mean"] = features["commentissuecos_sim"]
        features["commentissuejaccard_sim"] = self.calculate_text_similarity(
            user_commits_text, issue_text, "jaccard"
        )
        features["commentissuejaccard_sim_mean"] = features["commentissuejaccard_sim"]

        # Solved issue similarity (simplified)
        features["solvedisscos_sim"] = self.calculate_text_similarity(
            user_issues_text, issue_text, "cosine"
        )
        features["solvedisscos_mean"] = features["solvedisscos_sim"]
        features["solvedissjaccard_sim"] = self.calculate_text_similarity(
            user_issues_text, issue_text, "jaccard"
        )
        features["solvedissjaccard_sim_mean"] = features["solvedissjaccard_sim"]

        return features

    def collect_optimized_features_with_prefetched_data(
        self,
        username,
        owner,
        repo_name,
        issue_number,
        user_data=None,
        user_repos=None,
        user_commits=None,
        user_prs=None,
        repo_readme=None,
        repo_contributors=None,
        issue_data=None,
    ):
        """
        Collect optimized features using pre-fetched data to reduce API calls

        Args:
            username (str): GitHub username
            owner (str): Repository owner
            repo_name (str): Repository name
            issue_number (int): Issue number
            user_data (dict): Pre-fetched user data
            user_repos (list): Pre-fetched user repositories
            user_commits (list): Pre-fetched user commits
            user_prs (list): Pre-fetched user pull requests
            repo_readme (str): Pre-fetched repository README
            repo_contributors (list): Pre-fetched repository contributors
            issue_data (dict): Pre-fetched issue data

        Returns:
            dict: Optimized features
        """
        # Use pre-fetched data or fetch if not provided
        user_data = user_data or self.get_user_info(username)

        # Get issue data if not provided
        if not issue_data:
            issue_endpoint = f"repos/{owner}/{repo_name}/issues/{issue_number}"
            issue_data = self._make_request(issue_endpoint)
            if not issue_data:
                return None

        # Get repository info
        repo_info = self.get_repo_info(owner, repo_name)

        # Extract features...
        features = {}

        # Basic metadata
        features["username"] = username
        features["owner"] = owner
        features["repo"] = repo_name
        features["issue_number"] = issue_number

        # Extract user features
        user_features = self.extract_optimized_user_features(
            username, user_data, user_repos, user_commits, user_prs
        )
        features.update(user_features)

        # Extract issue features
        issue_features = self.extract_optimized_issue_features(issue_data)
        features.update(issue_features)

        # Extract repository features
        repo_features = self.extract_optimized_repo_features(
            owner, repo_name, repo_readme, repo_contributors, repo_info
        )
        features.update(repo_features)

        # Calculate similarity features between user and issue
        # Prepare data for similarity calculation
        user_data_for_sim = {
            "issues": user_prs or self.get_user_prs(username, state="all"),
            "prs": user_prs or self.get_user_prs(username, state="all"),
            "commits": user_commits or self.get_user_commits(username),
        }

        repo_data_for_sim = {
            "readme": repo_readme or self.get_repo_readme(owner, repo_name)
        }

        similarity_features = self.calculate_optimized_similarity_features(
            user_data_for_sim, issue_data, repo_data_for_sim
        )
        features.update(similarity_features)

        return features

    def collect_optimized_features(self, username, owner, repo_name, issue_number):
        """
        Collect only the important features for a user-issue pair based on feature importance analysis

        Args:
            username (str): GitHub username
            owner (str): Repository owner
            repo_name (str): Repository name
            issue_number (int): Issue number

        Returns:
            dict: Important features
        """
        print(
            f"Collecting optimized features for {username} on {owner}/{repo_name}#{issue_number}"
        )

        # Collect user data
        user_features = self.extract_optimized_user_features(username)

        # Collect repository data
        repo_features = self.extract_optimized_repo_features(owner, repo_name)

        # Get issue data
        issue_data = self._make_request(
            f"repos/{owner}/{repo_name}/issues/{issue_number}"
        )
        if not issue_data:
            return {}

        # Extract issue features
        issue_features = self.extract_optimized_issue_features(issue_data)

        # Get repository README
        readme = self.get_repo_readme(owner, repo_name)

        # Prepare data for similarity calculation
        user_data = {
            "issues": self.get_user_prs(username, state="all"),
            "prs": self.get_user_prs(username, state="all"),
            "commits": self.get_user_commits(username),
        }

        repo_data = {"readme": readme}

        # Calculate similarity features
        similarity_features = self.calculate_optimized_similarity_features(
            user_data, issue_data, repo_data
        )

        # Combine all features
        all_features = {}
        all_features.update(user_features)
        all_features.update(repo_features)
        all_features.update(issue_features)
        all_features.update(similarity_features)

        # Add metadata
        all_features["username"] = username
        all_features["owner"] = owner
        all_features["repo"] = repo_name
        all_features["issue_number"] = issue_number

        return all_features

    def collect_features_batch(self, usernames, owner, repo_name, issue_numbers):
        """
        Collect optimized features for multiple user-issue pairs

        Args:
            usernames (list): List of GitHub usernames
            owner (str): Repository owner
            repo_name (str): Repository name
            issue_numbers (list): List of issue numbers

        Returns:
            pd.DataFrame: Features for all user-issue pairs
        """
        all_features = []

        for i, username in enumerate(usernames):
            issue_number = issue_numbers[i]
            print(
                f"Collecting optimized features for {username} and issue #{issue_number} ({i+1}/{len(usernames)})"
            )

            features = self.collect_optimized_features(
                username, owner, repo_name, issue_number
            )
            if features:
                all_features.append(features)

        if not all_features:
            return pd.DataFrame()

        return pd.DataFrame(all_features)

    def save_features(self, features_df, filename):
        """
        Save features to a file

        Args:
            features_df (pd.DataFrame): Features dataframe
            filename (str): Output filename
        """
        # Save as pickle for preserving data types
        features_df.to_pickle(filename)
        print(f"Features saved to {filename}")

    def load_features(self, filename):
        """
        Load features from a file

        Args:
            filename (str): Input filename

        Returns:
            pd.DataFrame: Features dataframe
        """
        return pd.read_pickle(filename)


# Example usage
if __name__ == "__main__":
    # Check if token is provided
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Please set the GITHUB_TOKEN environment variable")
        sys.exit(1)

    # Create collector
    collector = OptimizedGitHubFeatureCollector(token)

    # Example: Collect features for a single user-issue pair
    username = "example_user"
    owner = "example_owner"
    repo_name = "example_repo"
    issue_number = 123

    features = collector.collect_optimized_features(
        username, owner, repo_name, issue_number
    )
    print(f"Collected {len(features)} features for {username} on issue #{issue_number}")

    # Convert to DataFrame and save
    features_df = pd.DataFrame([features])
    collector.save_features(features_df, "optimized_features_example.pkl")

    print("Done!")
