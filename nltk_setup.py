import os
import sys
import nltk
import shutil
from pathlib import Path


def setup_nltk():
    """Set up NLTK resources properly to avoid common errors"""

    # Create NLTK data directories
    nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)

    # Add to path
    nltk.data.path.append(nltk_data_dir)

    # Resources to download
    resources = ["punkt", "averaged_perceptron_tagger", "maxent_ne_chunker", "words"]

    for resource in resources:
        print(f"Downloading {resource}...")
        nltk.download(resource, download_dir=nltk_data_dir)

    # Specifically handle punkt_tab issue
    punkt_tab_dir = os.path.join(nltk_data_dir, "tokenizers", "punkt_tab", "english")
    os.makedirs(punkt_tab_dir, exist_ok=True)

    # Create empty files in punkt_tab to satisfy the lookup
    # This is a workaround as punkt_tab is sometimes needed but not available as a direct download
    Path(os.path.join(punkt_tab_dir, "punkt.pickle")).touch()

    # Test tokenization
    test_text = "Hello world. This is a test!"
    try:
        sentences = nltk.sent_tokenize(test_text)
        print(f"Tokenization test successful: {sentences}")
    except Exception as e:
        print(f"Tokenization test failed: {e}")

    # Print paths being searched
    print("\nNLTK data paths:")
    for path in nltk.data.path:
        print(f"- {path}")

    print("\nSetup complete!")


if __name__ == "__main__":
    setup_nltk()
