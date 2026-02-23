import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords and tokenizer if you haven't already
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('punkt_tab')

def load_data(file_path):
    """
    Load data from a file and return it as a list of lines.
    
    Args:
        file_path (str): The path to the file to be loaded.
    
    Returns:
        list: A list of lines from the file.
    """
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

def preprocess_data(data):
    """
    Preprocess the data by tokenizing the sentences and removing stop words.
    
    Args:
        data (list): A list of sentences to be preprocessed.
    Returns:
        list: A list of preprocessed sentences, where each sentence is a list of words.
    """    
    stop_words = set(stopwords.words("english"))
    preprocessed_data = []
    
    for sentence in data:
        # Tokenize the sentence into words
        words = word_tokenize(sentence.lower())
        
        # Remove stop words and non-alphabetic tokens
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
        
        # Only add non-empty sentences
        if filtered_words:
            preprocessed_data.append(filtered_words)
    
    return preprocessed_data