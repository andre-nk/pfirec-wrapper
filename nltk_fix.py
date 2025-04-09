import os
import sys
import nltk
from pathlib import Path
import shutil


def fix_nltk_tokenization():
    """
    Fix NLTK tokenization issues by properly setting up required resources
    """
    print("Setting up NLTK resources to fix tokenization issues...")

    # Create NLTK data directory
    nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)

    # Add directory to NLTK path
    nltk.data.path.insert(0, nltk_data_dir)

    # Download punkt tokenizer
    punkt_package = "punkt"
    print(f"Downloading {punkt_package}...")
    nltk.download(punkt_package, download_dir=nltk_data_dir, quiet=False)

    # Make sure punkt_tab directory structure exists
    punkt_tab_dir = os.path.join(nltk_data_dir, "tokenizers", "punkt_tab", "english")
    os.makedirs(punkt_tab_dir, exist_ok=True)

    # Copy punkt pickle file to punkt_tab directory
    punkt_src = os.path.join(nltk_data_dir, "tokenizers", "punkt", "english.pickle")
    punkt_dst = os.path.join(punkt_tab_dir, "punkt.pickle")

    if os.path.exists(punkt_src):
        print(f"Copying punkt data from {punkt_src} to {punkt_dst}")
        shutil.copy2(punkt_src, punkt_dst)
    else:
        print(f"Source file {punkt_src} not found, creating empty file at {punkt_dst}")
        Path(punkt_dst).touch()

    # Test the tokenizer
    test_text = "This is a test sentence. This is another test sentence."
    try:
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(test_text)
        print(f"✅ Tokenization test successful: {sentences}")
    except Exception as e:
        print(f"❌ Tokenization test failed: {e}")
        return False

    print("\nNLTK data paths:")
    for path in nltk.data.path:
        print(f"- {path}")

    print(
        "\n✅ NLTK tokenization setup complete. Future tokenization should work correctly."
    )
    return True


if __name__ == "__main__":
    success = fix_nltk_tokenization()
    sys.exit(0 if success else 1)
