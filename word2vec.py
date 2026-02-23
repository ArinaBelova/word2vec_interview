"""
Word2Vec model implementation using skip-gram architecture with negative sampling.
Implemented in pure NumPy without deep learning frameworks.
"""
import numpy as np


class Word2Vec:
    """
    A simple implementation of the Word2Vec model for learning word embeddings.
    Uses skip-gram architecture with negative sampling, implemented in pure NumPy.
    """
    def __init__(self, sentences, embedding_dim, window_size=2, learning_rate=0.01, negative_samples=5):
        """
        Initialize the Word2Vec model.
        
        Args:
            sentences (list): A list of sentences, where each sentence is a list of words.
            embedding_dim (int): The dimensionality of the word embeddings.
            window_size (int): The context window size for skip-gram.
            learning_rate (float): The learning rate for gradient descent.
            negative_samples (int): Number of negative samples per positive pair.
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.negative_samples = negative_samples
        self.sentences = sentences
        
        # Build vocabulary from sentences
        self.vocab = set()
        for sentence in sentences:
            for word in sentence:
                self.vocab.add(word)
        self.vocab = list(self.vocab)
        self.vocab_size = len(self.vocab)
        
        # Word to index and index to word mappings
        self._word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self._idx2word = {idx: word for word, idx in self._word2idx.items()}
        
        # Initialize weight matrices with small random values
        # W_embed: input word embeddings (vocab_size x embedding_dim)
        # W_context: context word embeddings (vocab_size x embedding_dim)
        self.W_embed = np.random.randn(self.vocab_size, embedding_dim) * 0.01
        self.W_context = np.random.randn(self.vocab_size, embedding_dim) * 0.01
        
        # Store word vectors after training
        self.word_to_vec = {}

    def _generate_training_pairs(self):
        """
        Generate (center_word, context_word) pairs from sentences using skip-gram.
        
        Returns:
            list: List of (center_idx, context_idx) tuples.
        """
        pairs = []
        for sentence in self.sentences:
            sentence_indices = [self._word2idx[word] for word in sentence]
            
            for i, center_idx in enumerate(sentence_indices):
                # Get context words within window (respecting sentence boundaries)
                start = max(0, i - self.window_size)
                end = min(len(sentence_indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:  # Skip the center word itself
                        context_idx = sentence_indices[j]
                        pairs.append((center_idx, context_idx))
        
        return pairs

    def _get_negative_samples(self, positive_idx):
        """
        Get random negative sample indices (words that are not the positive context).
        
        Args:
            positive_idx (int): The index of the positive context word to exclude.
        
        Returns:
            list: List of negative sample indices.
        """
        negative_indices = []
        while len(negative_indices) < self.negative_samples:
            neg_idx = np.random.randint(0, self.vocab_size)
            if neg_idx != positive_idx:
                negative_indices.append(neg_idx)
        return negative_indices

    def _sigmoid(self, x):
        """Compute sigmoid function with numerical stability."""
        return np.where(x >= 0, 
                        1 / (1 + np.exp(-x)), 
                        np.exp(x) / (1 + np.exp(x)))

    def _train_pair(self, center_idx, context_idx, label):
        """
        Train on a single (center, context) pair using negative sampling loss.
        
        Args:
            center_idx (int): Index of the center word.
            context_idx (int): Index of the context word.
            label (int): 1 for positive pair, 0 for negative pair.
        
        Returns:
            float: The loss for this pair.
        """
        # Forward pass
        center_vec = self.W_embed[center_idx]      # (embedding_dim,)
        context_vec = self.W_context[context_idx]  # (embedding_dim,)
        
        # Dot product and sigmoid
        dot_product = np.dot(center_vec, context_vec)
        prediction = self._sigmoid(dot_product)
        
        # Binary cross-entropy loss
        epsilon = 1e-7  # For numerical stability
        if label == 1:
            loss = -np.log(prediction + epsilon)
        else:
            loss = -np.log(1 - prediction + epsilon)
        
        # Backward pass: compute gradients
        # d(loss)/d(prediction) * d(prediction)/d(dot_product)
        grad_output = prediction - label  # Gradient of sigmoid + BCE combined
        
        # Gradients for embeddings
        grad_center = grad_output * context_vec    # (embedding_dim,)
        grad_context = grad_output * center_vec    # (embedding_dim,)
        
        # Update weights using gradient descent
        self.W_embed[center_idx] -= self.learning_rate * grad_center
        self.W_context[context_idx] -= self.learning_rate * grad_context
        
        return loss

    def word2idx(self):
        """
        Create a mapping from words to their corresponding indices in the embedding matrix.
        
        Returns:
            dict: A dictionary mapping words to their indices.
        """
        return self._word2idx
    
    def idx2word(self):
        """
        Create a mapping from indices to their corresponding words in the embedding matrix.
        
        Returns:
            dict: A dictionary mapping indices to their corresponding words.
        """
        return self._idx2word

    def get_embedding(self, word):
        """
        Get the embedding vector for a given word.
        
        Args:
            word (str): The word for which to retrieve the embedding.
        Returns:
            list: The embedding vector for the given word, or a zero vector if the word is not in the vocabulary.
        """
        return self.word_to_vec.get(word, [0] * self.embedding_dim)
    
    def most_similar(self, word, top_n=5):
        """
        Find the most similar words to a given word based on cosine similarity.
        
        Args:
            word (str): The word to find similar words for.
            top_n (int): Number of similar words to return.
        
        Returns:
            list: List of (word, similarity) tuples.
        """
        if word not in self._word2idx:
            return []
        
        word_vec = np.array(self.word_to_vec[word])
        word_norm = np.linalg.norm(word_vec)
        
        if word_norm == 0:
            return []
        
        similarities = []
        for other_word, other_vec in self.word_to_vec.items():
            if other_word != word:
                other_vec = np.array(other_vec)
                other_norm = np.linalg.norm(other_vec)
                if other_norm > 0:
                    similarity = np.dot(word_vec, other_vec) / (word_norm * other_norm)
                    similarities.append((other_word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]    