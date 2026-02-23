"""
Dataset utilities for Word2Vec training.
Provides functions to load various text corpora.
"""
import nltk
import os

# Download required NLTK data
nltk.download('brown', quiet=True)
nltk.download('reuters', quiet=True)
nltk.download('gutenberg', quiet=True)


def load_brown_corpus():
    """
    Load the Brown corpus - a balanced corpus of American English.
    Contains ~1 million words from 500 texts across 15 genres.
    
    Returns:
        list: List of sentences, each sentence is a list of lowercase words.
    """
    from nltk.corpus import brown
    sentences = []
    for sent in brown.sents():
        # Lowercase and filter alphabetic words only
        words = [word.lower() for word in sent if word.isalpha()]
        if len(words) >= 2:  # Need at least 2 words for context
            sentences.append(words)
    return sentences


def load_reuters_corpus():
    """
    Load the Reuters corpus - news articles.
    Contains ~1.3 million words.
    
    Returns:
        list: List of sentences, each sentence is a list of lowercase words.
    """
    from nltk.corpus import reuters
    sentences = []
    for fileid in reuters.fileids():
        for sent in reuters.sents(fileid):
            words = [word.lower() for word in sent if word.isalpha()]
            if len(words) >= 2:
                sentences.append(words)
    return sentences


def load_gutenberg_corpus():
    """
    Load the Gutenberg corpus - classic literature.
    Contains texts from Project Gutenberg.
    
    Returns:
        list: List of sentences, each sentence is a list of lowercase words.
    """
    from nltk.corpus import gutenberg
    sentences = []
    for fileid in gutenberg.fileids():
        for sent in gutenberg.sents(fileid):
            words = [word.lower() for word in sent if word.isalpha()]
            if len(words) >= 2:
                sentences.append(words)
    return sentences


def load_combined_corpus():
    """
    Load a combination of Brown, Reuters, and Gutenberg corpora.
    Provides a diverse and rich training set.
    
    Returns:
        list: List of sentences from all corpora combined.
    """
    print("Loading Brown corpus...")
    brown_sents = load_brown_corpus()
    print(f"  - {len(brown_sents)} sentences")
    
    print("Loading Reuters corpus...")
    reuters_sents = load_reuters_corpus()
    print(f"  - {len(reuters_sents)} sentences")
    
    print("Loading Gutenberg corpus...")
    gutenberg_sents = load_gutenberg_corpus()
    print(f"  - {len(gutenberg_sents)} sentences")
    
    all_sentences = brown_sents + reuters_sents + gutenberg_sents
    print(f"\nTotal: {len(all_sentences)} sentences")
    
    # Count vocabulary
    vocab = set(word for sent in all_sentences for word in sent)
    print(f"Vocabulary size: {len(vocab)} unique words")
    
    return all_sentences


def download_text8():
    """
    Download and extract the text8 dataset.
    Text8 is a cleaned Wikipedia dump (~100MB of text).
    
    Returns:
        str: Path to the text8 file.
    """
    import urllib.request
    import zipfile
    
    url = "http://mattmahoney.net/dc/text8.zip"
    zip_path = "text8.zip"
    text_path = "text8"
    
    if not os.path.exists(text_path):
        print("Downloading text8 dataset...")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        os.remove(zip_path)
        print("Done!")
    else:
        print("text8 already exists.")
    
    return text_path


def load_text8(max_sentences=None):
    """
    Load the text8 dataset, splitting into sentences.
    
    Args:
        max_sentences: Maximum number of sentences to load (None for all).
    
    Returns:
        list: List of sentences, each sentence is a list of words.
    """
    text_path = download_text8()
    
    with open(text_path, 'r') as f:
        text = f.read()
    
    # text8 is one long string, split into chunks as "sentences"
    words = text.split()
    sentence_length = 50  # Words per "sentence"
    
    sentences = []
    for i in range(0, len(words), sentence_length):
        sent = words[i:i + sentence_length]
        if len(sent) >= 2:
            sentences.append(sent)
            if max_sentences and len(sentences) >= max_sentences:
                break
    
    print(f"Loaded {len(sentences)} sentences from text8")
    return sentences


if __name__ == "__main__":
    # Demo: load combined corpus
    sentences = load_combined_corpus()
    
    # Show some examples
    print("\nSample sentences:")
    for i, sent in enumerate(sentences[:5]):
        print(f"  {i+1}. {' '.join(sent[:10])}...")
