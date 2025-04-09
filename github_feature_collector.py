import os
import time
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import re
from textblob import TextBlob
import math
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pickle

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load spaCy model for lemmatization
try:
    nlp = spacy.load('en_core_web_sm')
except:
    # If model not installed, download it
    import subprocess
    subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

class GitHubFeatureCollector:
    """
    Collects and constructs features for the PFIRec model from GitHub API
    """
    
    def __init__(self, token=None):
        """
        Initialize the collector with GitHub API token
        
        Args:
            token (str): GitHub API token
        """
        self.token = token or os.environ.get('GITHUB_TOKEN')
        if not self.token:
            raise ValueError("GitHub API token is required. Set GITHUB_TOKEN environment variable or pass token parameter.")
        
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = 0
        self.cache = {}
        
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
                print(f"Rate limit almost reached. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
        
        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)
        
        # Update rate limit info
        self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
        self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', 0))
        
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
            params = {'per_page': 100, 'page': page}
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
    
    def get_repo_issues(self, owner, repo, state='all'):
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
            params = {'per_page': 100, 'page': page, 'state': state}
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
            owner = repo['owner']['login']
            repo_name = repo['name']
            
            params = {'author': username, 'per_page': 100}
            if since:
                params['since'] = since.isoformat()
            
            repo_commits = self._make_request(f"repos/{owner}/{repo_name}/commits", params)
            if repo_commits:
                commits.extend(repo_commits)
        
        return commits
    
    def get_user_prs(self, username, state='all'):
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
        params = {'q': query, 'per_page': 100}
        
        prs = []
        page = 1
        while True:
            params['page'] = page
            search_results = self._make_request("search/issues", params)
            if not search_results or len(search_results.get('items', [])) == 0:
                break
            prs.extend(search_results['items'])
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
        if readme_data and 'content' in readme_data:
            import base64
            content = base64.b64decode(readme_data['content']).decode('utf-8')
            self.cache[cache_key] = content
            return content
        return ""
    
    def get_repo_languages(self, owner, repo):
        """
        Get repository languages
        
        Args:
            owner (str): Repository owner
            repo (str): Repository name
            
        Returns:
            dict: Language distribution
        """
        cache_key = f"languages_{owner}_{repo}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        languages = self._make_request(f"repos/{owner}/{repo}/languages")
        if languages:
            self.cache[cache_key] = languages
        return languages or {}
    
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
    
    def get_issue_labels(self, owner, repo, issue_number):
        """
        Get issue labels
        
        Args:
            owner (str): Repository owner
            repo (str): Repository name
            issue_number (int): Issue number
            
        Returns:
            list: Issue labels
        """
        cache_key = f"issue_labels_{owner}_{repo}_{issue_number}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        issue = self._make_request(f"repos/{owner}/{repo}/issues/{issue_number}")
        if issue and 'labels' in issue:
            labels = [label['name'] for label in issue['labels']]
            self.cache[cache_key] = labels
            return labels
        return []
    
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
    
    def calculate_euclidean_similarity(self, emb0, emb1):
        """
        Calculate similarity based on Euclidean distance
        
        Args:
            emb0 (array-like): First embedding
            emb1 (array-like): Second embedding
            
        Returns:
            float: Euclidean similarity score
        """
        try:
            return 1 / (1 + euclidean_distances(emb0, emb1)[0][0])
        except Exception as e:
            print(f"Error calculating Euclidean similarity: {e}")
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

    # def calculate_text_similarity(self, text1, text2, method='jaccard'):
    #     """
    #     Calculate similarity between two texts
        
    #     Args:
    #         text1 (str): First text
    #         text2 (str): Second text
    #         method (str): Similarity method (jaccard, cosine)
            
    #     Returns:
    #         float: Similarity score
    #     """
    #     if not text1 or not text2:
    #         return 0.0
        
    #     # Preprocess texts
    #     text1 = re.sub(r'[^\w\s]', '', text1.lower())
    #     text2 = re.sub(r'[^\w\s]', '', text2.lower())
        
    #     if method == 'jaccard':
    #         # Jaccard similarity
    #         set1 = set(text1.split())
    #         set2 = set(text2.split())
            
    #         if not set1 or not set2:
    #             return 0.0
                
    #         intersection = len(set1.intersection(set2))
    #         union = len(set1.union(set2))
            
    #         return intersection / union if union > 0 else 0.0
            
    #     elif method == 'cosine':
    #         # Cosine similarity
    #         vectorizer = TfidfVectorizer()
    #         try:
    #             tfidf_matrix = vectorizer.fit_transform([text1, text2])
    #             return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    #         except:
    #             return 0.0
        
    #     return 0.0
    
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
    
    def calculate_issue_sentiment_mean(self, titles, bodies):
        """
        Calculate mean sentiment for issue titles and bodies
        
        Args:
            titles (list): List of issue titles
            bodies (list): List of issue bodies
            
        Returns:
            float: Mean sentiment score
        """
        if not titles or not bodies:
            return 0.0
        
        try:
            sentiments = []
            for i in range(len(titles)):
                title = titles[i] or ""
                body = bodies[i] or ""
                text = title + ' ' + body
                sentiments.append(self.calculate_sentiment(text))
            
            return np.mean(sentiments) if sentiments else 0.0
        except Exception as e:
            print(f"Error calculating issue sentiment mean: {e}")
            return 0.0

    def calculate_readability(self, text):
        """
        Calculate readability metrics for text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Readability metrics
        """
        if not text:
            return {
                'coleman_liau_index': 0,
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'automated_readability_index': 0
            }
        
        # Count characters, words, and sentences
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = max(1, len(sent_tokenize(text)))
        
        # Average characters per word
        avg_chars_per_word = char_count / max(1, word_count)
        
        # Average words per sentence
        avg_words_per_sentence = word_count / max(1, sentence_count)
        
        # Coleman-Liau Index
        coleman_liau = 0.0588 * (avg_chars_per_word * 100) - 0.296 * (100 / avg_words_per_sentence) - 15.8
        
        # Flesch Reading Ease
        flesch_reading_ease = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_chars_per_word)
        
        # Flesch-Kincaid Grade Level
        flesch_kincaid_grade = 0.39 * avg_words_per_sentence + 11.8 * avg_chars_per_word - 15.59
        
        # Automated Readability Index
        automated_readability_index = 4.71 * avg_chars_per_word + 0.5 * avg_words_per_sentence - 21.43
        
        return {
            'coleman_liau_index': max(0, coleman_liau),
            'flesch_reading_ease': max(0, flesch_reading_ease),
            'flesch_kincaid_grade': max(0, flesch_kincaid_grade),
            'automated_readability_index': max(0, automated_readability_index)
        }
    
    def extract_issue_features(self, issue):
        """
        Extract features from an issue
        
        Args:
            issue (dict): Issue data
            
        Returns:
            dict: Issue features
        """
        features = {}
        
        # Basic issue information
        features['LengthOfTitle'] = len(issue.get('title', ''))
        features['LengthOfDescription'] = len(issue.get('body', '')) if issue.get('body') else 0
        
        # Count code blocks, URLs, and images in body
        body = issue.get('body', '') or ''
        features['NumOfCode'] = body.count('```')
        features['NumOfUrls'] = len(re.findall(r'https?://\S+', body))
        features['NumOfPics'] = len(re.findall(r'!\[.*?\]\(.*?\)', body)) + body.count('<img')
        
        # Sentiment
        features['issuesen'] = self.calculate_sentiment(issue.get('title', '') + ' ' + (issue.get('body', '') or ''))
        
        # Readability
        readability = self.calculate_readability(issue.get('title', '') + ' ' + (issue.get('body', '') or ''))
        features.update(readability)
        
        # Labels
        labels = [label['name'].lower() for label in issue.get('labels', [])]
        features['labelnum'] = len(labels)
        
        # Specific label types
        features['buglabelnum'] = sum(1 for label in labels if 'bug' in label)
        features['featurelabelnum'] = sum(1 for label in labels if 'feature' in label or 'enhancement' in label)
        features['testlabelnum'] = sum(1 for label in labels if 'test' in label)
        features['buildlabelnum'] = sum(1 for label in labels if 'build' in label)
        features['doclabelnum'] = sum(1 for label in labels if 'doc' in label)
        features['codinglabelnum'] = sum(1 for label in labels if 'code' in label)
        features['enhancelabelnum'] = sum(1 for label in labels if 'enhance' in label)
        features['gfilabelnum'] = sum(1 for label in labels if 'good first issue' in label or 'good-first-issue' in label)
        features['mediumlabelnum'] = sum(1 for label in labels if 'medium' in label)
        features['majorlabelnum'] = sum(1 for label in labels if 'major' in label)
        features['triagedlabelnum'] = sum(1 for label in labels if 'triaged' in label)
        features['untriagedlabelnum'] = sum(1 for label in labels if 'untriaged' in label)
        
        return features
    
    def extract_repo_features(self, owner, repo_name):
        """
        Extract features from a repository
        
        Args:
            owner (str): Repository owner
            repo_name (str): Repository name
            
        Returns:
            dict: Repository features
        """
        features = {}
        
        # Get repository information
        repo_info = self.get_repo_info(owner, repo_name)
        if not repo_info:
            return features
        
        # Basic repository metrics
        features['pro_star'] = repo_info.get('stargazers_count', 0)
        features['procmt'] = repo_info.get('size', 0)  # Size is a rough proxy for commits
        
        # Issues
        issues = self.get_repo_issues(owner, repo_name)
        open_issues = [issue for issue in issues if issue.get('state') == 'open']
        features['openiss'] = len(open_issues)
        features['openissratio'] = len(open_issues) / max(1, len(issues)) if issues else 0
        
        # Good First Issues
        gfi_issues = [issue for issue in issues if any('good first issue' in label['name'].lower() or 'good-first-issue' in label['name'].lower() for label in issue.get('labels', []))]
        features['pro_gfi_num'] = len(gfi_issues)
        features['pro_gfi_ratio'] = len(gfi_issues) / max(1, len(issues)) if issues else 0
        
        # Contributors
        contributors = self.get_repo_contributors(owner, repo_name)
        features['contributornum'] = len(contributors)
        
        # Critical issues (approximation)
        critical_issues = [issue for issue in issues if any('critical' in label['name'].lower() or 'high' in label['name'].lower() for label in issue.get('labels', []))]
        features['crtclsissnum'] = len(critical_issues)
        
        # Pull requests (approximation)
        features['proclspr'] = repo_info.get('open_issues_count', 0) - len(open_issues)
        
        return features
    
    def extract_user_features(self, username):
        """
        Extract features for a user
        
        Args:
            username (str): GitHub username
            
        Returns:
            dict: User features
        """
        features = {}
        
        # Get user information
        user_info = self.get_user_info(username)
        if not user_info:
            return features
        
        # Get user repositories
        user_repos = self.get_user_repos(username)
        
        # Get user commits (last 3 months)
        three_months_ago = datetime.now() - timedelta(days=90)
        commits = self.get_user_commits(username, since=three_months_ago)
        
        # Get user PRs
        prs = self.get_user_prs(username)
        
        # Get user issues
        issues = []
        for repo in user_repos:
            repo_issues = self.get_repo_issues(repo['owner']['login'], repo['name'])
            user_issues = [issue for issue in repo_issues if issue.get('user', {}).get('login') == username]
            issues.extend(user_issues)
        
        # General OSS experience
        features['clsallcmt'] = len(commits)
        features['clsallpr'] = len(prs)
        features['clsalliss'] = len(issues)
        features['clspronum'] = len(user_repos)
        features['clsiss'] = len(issues)
        features['clsallprreview'] = 0  # Need PR review API
        
        # Activeness (by month)
        one_month_ago = datetime.now() - timedelta(days=30)
        two_months_ago = datetime.now() - timedelta(days=60)
        
        # Commits by month
        commit_dates = [datetime.strptime(commit['commit']['author']['date'], '%Y-%m-%dT%H:%M:%SZ') for commit in commits if 'commit' in commit and 'author' in commit['commit'] and 'date' in commit['commit']['author']]
        features['clsonemonth_cmt'] = sum(1 for date in commit_dates if date >= one_month_ago)
        features['clstwomonth_cmt'] = sum(1 for date in commit_dates if date >= two_months_ago and date < one_month_ago)
        features['clsthreemonth_cmt'] = sum(1 for date in commit_dates if date >= three_months_ago and date < two_months_ago)
        
        # PRs by month
        pr_dates = [datetime.strptime(pr['created_at'], '%Y-%m-%dT%H:%M:%SZ') for pr in prs if 'created_at' in pr]
        features['clsonemonth_pr'] = sum(1 for date in pr_dates if date >= one_month_ago)
        features['clstwomonth_pr'] = sum(1 for date in pr_dates if date >= two_months_ago and date < one_month_ago)
        features['clsthreemonth_pr'] = sum(1 for date in pr_dates if date >= three_months_ago and date < two_months_ago)
        
        # Issues by month
        issue_dates = [datetime.strptime(issue['created_at'], '%Y-%m-%dT%H:%M:%SZ') for issue in issues if 'created_at' in issue]
        features['clsonemonth_iss'] = sum(1 for date in issue_dates if date >= one_month_ago)
        features['clstwomonth_iss'] = sum(1 for date in issue_dates if date >= two_months_ago and date < one_month_ago)
        features['clsthreemonth_iss'] = sum(1 for date in issue_dates if date >= three_months_ago and date < two_months_ago)
        
        # Sentiment analysis
        issue_sentiments = [self.calculate_sentiment(issue.get('title', '') + ' ' + (issue.get('body', '') or '')) for issue in issues]
        pr_sentiments = [self.calculate_sentiment(pr.get('title', '') + ' ' + (pr.get('body', '') or '')) for pr in prs]
        
        features['clsissuesenmean'] = np.mean(issue_sentiments) if issue_sentiments else 0
        features['clsissuesenmedian'] = np.median(issue_sentiments) if issue_sentiments else 0
        features['clsprsenmean'] = np.mean(pr_sentiments) if pr_sentiments else 0
        features['clsprsenmedian'] = np.median(pr_sentiments) if pr_sentiments else 0
        
        return features
    
    def calculate_similarity_features(self, user_data, issue_data, repo_data):
        """
        Calculate similarity features between user and issue/repo
        
        Args:
            user_data (dict): User data
            issue_data (dict): Issue data
            repo_data (dict): Repository data
            
        Returns:
            dict: Similarity features
        """
        features = {}
        
        # Get user issues, PRs, commits text
        user_issues_text = ' '.join([issue.get('title', '') + ' ' + (issue.get('body', '') or '') for issue in user_data.get('issues', [])])
        user_prs_text = ' '.join([pr.get('title', '') + ' ' + (pr.get('body', '') or '') for pr in user_data.get('prs', [])])
        user_commits_text = ' '.join([commit.get('commit', {}).get('message', '') for commit in user_data.get('commits', [])])
        
        # Get issue text
        issue_text = issue_data.get('title', '') + ' ' + (issue_data.get('body', '') or '')
        
        # Get repo text (README)
        repo_readme = repo_data.get('readme', '')
        
        # Calculate similarities
        features['issjaccard_sim'] = self.calculate_text_similarity(user_issues_text, issue_text, 'jaccard')
        features['isscos_sim'] = self.calculate_text_similarity(user_issues_text, issue_text, 'cosine')
        
        features['cmtjaccard_sim'] = self.calculate_text_similarity(user_commits_text, issue_text, 'jaccard')
        features['cmtcos_sim'] = self.calculate_text_similarity(user_commits_text, issue_text, 'cosine')
        
        features['prjaccard_sim'] = self.calculate_text_similarity(user_prs_text, issue_text, 'jaccard')
        features['prcos_sim'] = self.calculate_text_similarity(user_prs_text, issue_text, 'cosine')
        
        features['readmejaccard_sim'] = self.calculate_text_similarity(user_issues_text + user_prs_text + user_commits_text, repo_readme, 'jaccard')
        features['readmecos_sim'] = self.calculate_text_similarity(user_issues_text + user_prs_text + user_commits_text, repo_readme, 'cosine')
        
        # Calculate mean similarities (simplified)
        features['issjaccard_sim_mean'] = features['issjaccard_sim']
        features['isscos_sim_mean'] = features['isscos_sim']
        features['cmtjaccard_sim_mean'] = features['cmtjaccard_sim']
        features['cmtcos_sim_mean'] = features['cmtcos_sim']
        features['prjaccard_sim_mean'] = features['prjaccard_sim']
        features['prcos_sim_mean'] = features['prcos_sim']
        features['readmejaccard_sim_mean'] = features['readmejaccard_sim']
        features['readmecos_sim_mean'] = features['readmecos_sim']
        
        # Language similarity (simplified)
        features['lan_sim'] = 1.0  # Default to 1.0 for now
        
        return features
    
    def collect_all_features(self, username, owner, repo_name, issue_number):
        """
        Collect all features for a user-issue pair
        
        Args:
            username (str): GitHub username
            owner (str): Repository owner
            repo_name (str): Repository name
            issue_number (int): Issue number
            
        Returns:
            dict: All features
        """
        # Collect user data
        user_features = self.extract_user_features(username)
        
        # Collect repository data
        repo_features = self.extract_repo_features(owner, repo_name)
        
        # Get issue data
        issue_data = self._make_request(f"repos/{owner}/{repo_name}/issues/{issue_number}")
        if not issue_data:
            return {}
        
        # Extract issue features
        issue_features = self.extract_issue_features(issue_data)
        
        # Get repository README
        readme = self.get_repo_readme(owner, repo_name)
        
        # Prepare data for similarity calculation
        user_data = {
            'issues': self.get_user_prs(username, state='all'),
            'prs': self.get_user_prs(username, state='all'),
            'commits': self.get_user_commits(username)
        }
        
        repo_data = {
            'readme': readme
        }
        
        # Calculate similarity features
        similarity_features = self.calculate_similarity_features(user_data, issue_data, repo_data)
        
        # Combine all features
        all_features = {}
        all_features.update(user_features)
        all_features.update(repo_features)
        all_features.update(issue_features)
        all_features.update(similarity_features)
        
        # Fill missing features with defaults
        self._fill_missing_features(all_features)
        
        return all_features
    
    def _fill_missing_features(self, features):
        """
        Fill missing features with default values
        
        Args:
            features (dict): Feature dictionary
            
        Returns:
            dict: Updated features
        """
        # Define all expected features and their default values
        default_features = {
            # User features
            "clsallcmt": 0, "clsallpr": 0, "clsalliss": 0, "clspronum": 0, "clsiss": 0, "clsallprreview": 0,
            "clsonemonth_cmt": 0, "clstwomonth_cmt": 0, "clsthreemonth_cmt": 0,
            "clsonemonth_pr": 0, "clstwomonth_pr": 0, "clsthreemonth_pr": 0,
            "clsonemonth_iss": 0, "clstwomonth_iss": 0, "clsthreemonth_iss": 0,
            "clsissuesenmean": 0, "clsissuesenmedian": 0, "clsprsenmean": 0, "clsprsenmedian": 0,
            
            # Similarity features
            "solvedisscos_sim": 0, "solvedisscos_mean": 0,
            "solvedissjaccard_sim": 0, "solvedissjaccard_sim_mean": 0,
            "solvedissuelabel_sum": 0, "solvedissuelabel_ratio": 0,
            "issjaccard_sim": 0, "issjaccard_sim_mean": 0,
            "isscos_sim": 0, "isscos_mean": 0,
            "issuelabel_sum": 0, "issuelabel_ratio": 0,
            "commentissuelabel_sum": 0, "commentissuelabel_ratio": 0,
            "commentissuecos_sim": 0, "commentissuecos_sim_mean": 0,
            "commentissuejaccard_sim": 0, "commentissuejaccard_sim_mean": 0,
            "cmtjaccard_sim": 0, "cmtjaccard_sim_mean": 0,
            "cmtcos_sim": 0, "cmtcos_mean": 0,
            "prjaccard_sim": 0, "prjaccard_sim_mean": 0,
            "prcos_sim": 0, "prcos_mean": 0,
            "prlabel_sum": 0, "prlabel_ratio": 0,
            "prreviewcos_sim": 0, "prreviewcos_sim_mean": 0,
            "prreviewjaccard_sim": 0, "prreviewjaccard_sim_mean": 0,
            "lan_sim": 0,
            
            # Domain features
            "readmecos_sim_mean": 0, "readmecos_sim": 0,
            "readmejaccard_sim_mean": 0, "readmejaccard_sim": 0,
            "procos_mean": 0, "procos_sim": 0,
            "projaccard_mean": 0, "projaccard_sim": 0,
            "prostopic_sum": 0, "prostopic_ratio": 0,
            
            # Issue content features
            "LengthOfTitle": 0, "LengthOfDescription": 0, "NumOfCode": 0, "NumOfUrls": 0, "NumOfPics": 0,
            "buglabelnum": 0, "featurelabelnum": 0, "testlabelnum": 0, "buildlabelnum": 0, "doclabelnum": 0,
            "codinglabelnum": 0, "enhancelabelnum": 0, "gfilabelnum": 0, "mediumlabelnum": 0, "majorlabelnum": 0,
            "triagedlabelnum": 0, "untriagedlabelnum": 0, "labelnum": 0,
            "issuesen": 0, "coleman_liau_index": 0, "flesch_reading_ease": 0, "flesch_kincaid_grade": 0,
            "automated_readability_index": 0,
            
            # Background features
            "pro_gfi_ratio": 0, "pro_gfi_num": 0, "proclspr": 0, "crtclsissnum": 0, "pro_star": 0,
            "openiss": 0, "openissratio": 0, "contributornum": 0, "procmt": 0,
            "rptcmt": 0, "rptiss": 0, "rptpr": 0, "rptpronum": 0, "rptallcmt": 0, "rptalliss": 0, "rptallpr": 0,
            "rpt_reviews_num_all": 0, "rpt_max_stars_commit": 0, "rpt_max_stars_issue": 0, "rpt_max_stars_pull": 0,
            "rpt_max_stars_review": 0, "rptisnew": 0, "rpt_gfi_ratio": 0,
            "ownercmt": 0, "owneriss": 0, "ownerpr": 0, "ownerpronum": 0, "ownerallcmt": 0, "owneralliss": 0,
            "ownerallpr": 0, "owner_reviews_num_all": 0, "owner_max_stars_commit": 0, "owner_max_stars_issue": 0,
            "owner_max_stars_pull": 0, "owner_max_stars_review": 0, "owner_gfi_ratio": 0, "owner_gfi_num": 0
        }
        
        # Add missing features
        for feature, default_value in default_features.items():
            if feature not in features:
                features[feature] = default_value
                    
        return features
    
    def collect_features_batch(self, usernames, owner, repo_name, issue_numbers):
        """
        Collect features for multiple user-issue pairs
        
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
            print(f"Collecting features for {username} and issue #{issue_number} ({i+1}/{len(usernames)})")
            
            features = self.collect_all_features(username, owner, repo_name, issue_number)
            if features:
                features['username'] = username
                features['owner'] = owner
                features['name'] = repo_name
                features['number'] = issue_number
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


class PFIRecFeatureProcessor:
    """
    Process and prepare features for the PFIRec model
    """
    
    def __init__(self):
        """Initialize the processor"""
        pass
    
    def prepare_features_for_prediction(self, features_df):
        """
        Prepare features for prediction
        
        Args:
            features_df (pd.DataFrame): Raw features
            
        Returns:
            pd.DataFrame: Processed features ready for model input
        """
        # Define all feature groups as in the original model
        xnames_sub_cumu = [
            "clsallcmt", "clsallpr", "clsalliss", "clspronum", "clsiss", 'clsallprreview',
        ]
        
        xnames_sub_act = [
            'clsonemonth_cmt', 'clstwomonth_cmt', 'clsthreemonth_cmt', 
            'clsonemonth_pr', 'clstwomonth_pr', 'clsthreemonth_pr', 
            'clsonemonth_iss', 'clstwomonth_iss', 'clsthreemonth_iss',
        ]
        
        xnames_sub_sen = [
            'clsissuesenmean', 'clsissuesenmedian', 'clsprsenmean', 'clsprsenmedian',
        ]
        
        xnames_sub_clssolvediss = [
            'solvedisscos_sim', 'solvedisscos_mean',
            "solvedissjaccard_sim", "solvedissjaccard_sim_mean",
            "solvedissuelabel_sum", "solvedissuelabel_ratio",
        ]
        
        xnames_sub_clsrptiss = [
            'issjaccard_sim', 'issjaccard_sim_mean',
            "isscos_sim", "isscos_mean",
            'issuelabel_sum', 'issuelabel_ratio',
        ]
        
        xnames_sub_clscomtiss = [
            'commentissuelabel_sum', 'commentissuelabel_ratio',
            'commentissuecos_sim', 'commentissuecos_sim_mean',
            'commentissuejaccard_sim', 'commentissuejaccard_sim_mean',
        ]
        
        xnames_sub_clscmt = [
            'cmtjaccard_sim', 'cmtjaccard_sim_mean',
            "cmtcos_sim", "cmtcos_mean",
        ]
        
        xnames_sub_clspr = [
            "prjaccard_sim", 'prjaccard_sim_mean',
            "prcos_sim", "prcos_mean",
            'prlabel_sum', 'prlabel_ratio',
        ]
        
        xnames_sub_clsprreview = [
            'prreviewcos_sim', 'prreviewcos_sim_mean',
            'prreviewjaccard_sim', 'prreviewjaccard_sim_mean',
        ]
        
        xnames_sub_clscont = xnames_sub_clscmt + xnames_sub_clspr + xnames_sub_clsprreview + \
                            xnames_sub_clsrptiss + xnames_sub_clscomtiss + xnames_sub_clssolvediss + ['lan_sim']
        
        xnames_sub_domain = [
            'readmecos_sim_mean', 'readmecos_sim',
            'readmejaccard_sim_mean', 'readmejaccard_sim',
            "procos_mean", "procos_sim",
            "projaccard_mean", 'projaccard_sim',
            'prostopic_sum', 'prostopic_ratio',
        ]
        
        xnames_sub_isscont = [
            'LengthOfTitle', 'LengthOfDescription', 'NumOfCode', 'NumOfUrls', 'NumOfPics', 
            'buglabelnum', 'featurelabelnum', 'testlabelnum', 'buildlabelnum', 'doclabelnum', 
            'codinglabelnum', 'enhancelabelnum', 'gfilabelnum', 'mediumlabelnum', 'majorlabelnum', 
            'triagedlabelnum', 'untriagedlabelnum', 'labelnum',
            'issuesen', 'coleman_liau_index', 'flesch_reading_ease', 'flesch_kincaid_grade', 'automated_readability_index',
        ]
        
        xnames_sub_back = [
            'pro_gfi_ratio', 'pro_gfi_num', 'proclspr', 'crtclsissnum', 'pro_star', 'openiss', 'openissratio', 'contributornum', 'procmt',
            'rptcmt', 'rptiss', 'rptpr', 'rptpronum', 'rptallcmt', 'rptalliss', 'rptallpr', 'rpt_reviews_num_all', 
            'rpt_max_stars_commit', 'rpt_max_stars_issue', 'rpt_max_stars_pull', 'rpt_max_stars_review', 'rptisnew', 'rpt_gfi_ratio',
            'ownercmt', 'owneriss', 'ownerpr', 'ownerpronum', 'ownerallcmt', 'owneralliss', 'ownerallpr', 'owner_reviews_num_all', 
            'owner_max_stars_commit', 'owner_max_stars_issue', 'owner_max_stars_pull', 'owner_max_stars_review', 'owner_gfi_ratio', 'owner_gfi_num',
        ]
        
        # Combine all features
        xnames_LambdaMART = xnames_sub_cumu + xnames_sub_act + xnames_sub_sen + xnames_sub_clscont + \
                           xnames_sub_domain + xnames_sub_isscont + xnames_sub_back
        
        # Ensure all required features are present
        for feature in xnames_LambdaMART:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        # Return only the features needed for the model
        return features_df[xnames_LambdaMART]
    
    def add_group_id(self, features_df, group_column='username'):
        """
        Add group ID for LambdaMART ranking
        
        Args:
            features_df (pd.DataFrame): Features dataframe
            group_column (str): Column to group by
            
        Returns:
            pd.DataFrame: Features with group ID
        """
        # Create a copy to avoid modifying the original
        df = features_df.copy()
        
        # Create a mapping from group values to IDs
        unique_groups = df[group_column].unique()
        group_to_id = {group: i for i, group in enumerate(unique_groups)}
        
        # Add the group ID column
        df['issgroupid'] = df[group_column].map(group_to_id)
        
        return df


def main():
    """
    Main function to demonstrate usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect features for PFIRec model')
    parser.add_argument('--token', type=str, help='GitHub API token')
    parser.add_argument('--owner', type=str, help='Repository owner')
    parser.add_argument('--repo', type=str, help='Repository name')
    parser.add_argument('--issue', type=int, help='Issue number')
    parser.add_argument('--user', type=str, help='GitHub username')
    parser.add_argument('--output', type=str, default='features.pkl', help='Output file')
    
    args = parser.parse_args()
    
    # Use environment variable if token not provided
    token = args.token or os.environ.get('GITHUB_TOKEN')
    
    if not token:
        print("GitHub API token is required. Set GITHUB_TOKEN environment variable or use --token.")
        return
    
    if not args.owner or not args.repo or not args.issue or not args.user:
        print("Repository owner, name, issue number, and username are required.")
        return
    
    # Initialize collector
    collector = GitHubFeatureCollector(token)
    
    # Collect features
    features = collector.collect_all_features(args.user, args.owner, args.repo, args.issue)
    
    if not features:
        print("Failed to collect features.")
        return
    
    # Add metadata
    features['username'] = args.user
    features['owner'] = args.owner
    features['name'] = args.repo
    features['number'] = args.issue
    
    # Create DataFrame
    features_df = pd.DataFrame([features])
    
    # Save features
    collector.save_features(features_df, args.output)
    
    # Process features for model input
    processor = PFIRecFeatureProcessor()
    model_features = processor.prepare_features_for_prediction(features_df)
    
    print(f"Collected {len(model_features.columns)} features for {args.user} and issue #{args.issue}")
    print(f"Features saved to {args.output}")


if __name__ == "__main__":
    main()