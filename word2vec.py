"""
Word2Vec model implementation using skip-gram architecture with negative sampling.
Implemented in pure NumPy without deep learning frameworks.
Optimized with batching, subsampling, vectorized operations, and Numba JIT.
"""
import numpy as np
from collections import Counter
from numba import njit, prange
import os
import pickle


@njit(parallel=True, fastmath=True)
def _train_batch_numba(W_embed, W_context, center_indices, context_indices, labels, learning_rate):
    """
    Numba-JIT compiled batch training function.
    Runs in parallel across CPU cores.
    """
    batch_size = len(center_indices)
    embedding_dim = W_embed.shape[1]
    total_loss = 0.0
    
    for i in prange(batch_size):
        center_idx = center_indices[i]
        context_idx = context_indices[i]
        label = labels[i]
        
        # Get vectors
        center_vec = W_embed[center_idx]
        context_vec = W_context[context_idx]
        
        # Dot product
        dot = 0.0
        for d in range(embedding_dim):
            dot += center_vec[d] * context_vec[d]
        
        # Sigmoid with numerical stability
        if dot >= 0:
            sig = 1.0 / (1.0 + np.exp(-dot))
        else:
            exp_dot = np.exp(dot)
            sig = exp_dot / (1.0 + exp_dot)
        
        # BCE loss
        if label == 1:
            total_loss += -np.log(sig + 1e-7)
        else:
            total_loss += -np.log(1 - sig + 1e-7)
        
        # Gradient
        grad = sig - label
        
        # Update weights
        for d in range(embedding_dim):
            grad_center = grad * context_vec[d]
            grad_context = grad * center_vec[d]
            W_embed[center_idx, d] -= learning_rate * grad_center
            W_context[context_idx, d] -= learning_rate * grad_context
    
    return total_loss


class Word2Vec:
    """
    A simple implementation of the Word2Vec model for learning word embeddings.
    Uses skip-gram architecture with negative sampling, implemented in pure NumPy.
    """
    def __init__(self, sentences, embedding_dim, window_size=2, learning_rate=0.01, 
                 negative_samples=5, subsample_threshold=1e-3, batch_size=256):
        """
        Initialize the Word2Vec model.
        
        Args:
            sentences (list): A list of sentences, where each sentence is a list of words.
            embedding_dim (int): The dimensionality of the word embeddings.
            window_size (int): The context window size for skip-gram.
            learning_rate (float): The learning rate for gradient descent.
            negative_samples (int): Number of negative samples per positive pair.
            subsample_threshold (float): Threshold for subsampling frequent words.
            batch_size (int): Number of word pairs to process in each batch.
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.negative_samples = negative_samples
        self.subsample_threshold = subsample_threshold
        self.batch_size = batch_size
        self.sentences = sentences
        
        # Count word frequencies
        word_counts = Counter()
        total_words = 0
        for sentence in sentences:
            for word in sentence:
                word_counts[word] += 1
                total_words += 1
        
        # Build vocabulary from sentences
        self.vocab = list(word_counts.keys())
        self.vocab_size = len(self.vocab)
        
        # Word to index and index to word mappings
        self._word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self._idx2word = {idx: word for word, idx in self._word2idx.items()}
        
        # Store word frequencies for subsampling and negative sampling
        self.word_freqs = np.array([word_counts[w] for w in self.vocab], dtype=np.float64)
        self.word_probs = self.word_freqs / total_words
        
        # Compute subsampling probabilities (discard probability for frequent words)
        # P(discard) = 1 - sqrt(t / f(w)) where t is threshold, f(w) is word frequency
        # If threshold is 0 or negative, disable subsampling entirely
        if self.subsample_threshold > 0:
            self.subsample_probs = 1.0 - np.sqrt(self.subsample_threshold / (self.word_probs + 1e-10))
            self.subsample_probs = np.clip(self.subsample_probs, 0, 1)
        else:
            # Keep all words
            self.subsample_probs = np.zeros(self.vocab_size)  
        
        # Compute negative sampling distribution (unigram^0.75)
        # heruistics derived in word2vec paper as explained here: https://arxiv.org/pdf/1310.4546.pdf (section 3.2.1)
        self.neg_sample_probs = np.power(self.word_freqs, 0.75)
        self.neg_sample_probs /= self.neg_sample_probs.sum()
        
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
        Applies subsampling to discard frequent words.
        
        Returns:
            np.ndarray: Array of shape (N, 2) with (center_idx, context_idx) pairs.
        """
        pairs = []
        for sentence in self.sentences:
            # Convert to indices and apply subsampling
            sentence_indices = []
            for word in sentence:
                idx = self._word2idx[word]
                # Keep word with probability (1 - subsample_prob)
                if np.random.random() > self.subsample_probs[idx]:
                    sentence_indices.append(idx)
            
            if len(sentence_indices) < 2:
                continue
                
            for i, center_idx in enumerate(sentence_indices):
                # Get context words within window (respecting sentence boundaries)
                start = max(0, i - self.window_size)
                end = min(len(sentence_indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:  # Skip the center word itself
                        context_idx = sentence_indices[j]
                        pairs.append((center_idx, context_idx))
        
        return np.array(pairs, dtype=np.int32) if pairs else np.array([], dtype=np.int32)

    def _get_negative_samples_batch(self, positive_indices, batch_size):
        """
        Get negative samples for a batch using unigram^0.75 distribution.
        Ensures negative samples don't include the positive context word.
        
        Args:
            positive_indices (np.ndarray): Array of positive context indices to exclude.
            batch_size (int): Number of samples in the batch.
        
        Returns:
            np.ndarray: Array of shape (batch_size, negative_samples) with negative indices.
        """
        # Sample all at once (vectorized)
        neg_samples = np.random.choice(
            self.vocab_size,
            size=(batch_size, self.negative_samples),
            p=self.neg_sample_probs
        )
        
        # Find collisions with positive indices and resample them
        # For large vocab, collisions are rare (~0.0025% per sample)
        positive_indices_expanded = positive_indices[:, np.newaxis]  # (batch_size, 1)
        collisions = (neg_samples == positive_indices_expanded)  # (batch_size, neg_samples)
        
        # Resample only the collisions (usually very few)
        collision_indices = np.where(collisions)
        for i, j in zip(*collision_indices):
            while neg_samples[i, j] == positive_indices[i]:
                neg_samples[i, j] = np.random.choice(self.vocab_size, p=self.neg_sample_probs)
        
        return neg_samples

    def _sigmoid(self, x):
        """Compute sigmoid function with numerical stability."""
        x = np.clip(x, -500, 500)  # Prevent overflow
        return np.where(x >= 0, 
                        1 / (1 + np.exp(-x)), 
                        np.exp(x) / (1 + np.exp(x)))

    def _train_batch(self, center_indices, context_indices, labels, use_numba=True):
        """
        Train on a batch of (center, context) pairs.
        
        Args:
            center_indices (np.ndarray): Array of center word indices (batch_size,).
            context_indices (np.ndarray): Array of context word indices (batch_size,).
            labels (np.ndarray): Array of labels, 1 for positive, 0 for negative (batch_size,).
            use_numba (bool): Whether to use Numba JIT compilation for speedup.
        
        Returns:
            float: Total loss for the batch.
        """
        if use_numba and len(center_indices) > 0:
            # Use Numba-compiled parallel function
            return _train_batch_numba(
                self.W_embed, self.W_context,
                center_indices.astype(np.int64),
                context_indices.astype(np.int64),
                labels.astype(np.float64),
                self.learning_rate
            )
        
        # Fallback to pure NumPy (for compatibility)
        center_vecs = self.W_embed[center_indices]
        context_vecs = self.W_context[context_indices]
        
        dot_products = np.sum(center_vecs * context_vecs, axis=1)
        predictions = self._sigmoid(dot_products)
        
        epsilon = 1e-7
        loss = -np.sum(labels * np.log(predictions + epsilon) + 
                       (1 - labels) * np.log(1 - predictions + epsilon))
        
        grad_output = (predictions - labels).reshape(-1, 1)
        grad_center = grad_output * context_vecs
        grad_context = grad_output * center_vecs
        
        np.add.at(self.W_embed, center_indices, -self.learning_rate * grad_center)
        np.add.at(self.W_context, context_indices, -self.learning_rate * grad_context)
        
        return loss

    def _train_pair(self, center_idx, context_idx, label):
        """
        Train on a single (center, context) pair using negative sampling loss.
        Kept for backward compatibility with tests.
        
        Args:
            center_idx (int): Index of the center word.
            context_idx (int): Index of the context word.
            label (int): 1 for positive pair, 0 for negative pair.
        
        Returns:
            float: The loss for this pair.
        """
        return self._train_batch(
            np.array([center_idx]), 
            np.array([context_idx]), 
            np.array([label])
        )

    def _get_negative_samples(self, positive_idx):
        """
        Get random negative sample indices (words that are not the positive context).
        Kept for backward compatibility with tests.
        
        Args:
            positive_idx (int): The index of the positive context word to exclude.
        
        Returns:
            list: List of negative sample indices.
        """
        samples = []
        while len(samples) < self.negative_samples:
            candidates = np.random.choice(
                self.vocab_size,
                size=self.negative_samples - len(samples),
                p=self.neg_sample_probs
            )
            for c in candidates:
                if c != positive_idx and len(samples) < self.negative_samples:
                    samples.append(c)
        return samples

    def train_epoch(self, training_pairs):
        """
        Train one epoch using batched processing.
        Parallelization is handled by Numba JIT via set_num_threads().
        
        Args:
            training_pairs (np.ndarray): Array of (center_idx, context_idx) pairs.
        
        Returns:
            float: Average loss for the epoch.
        """
        if len(training_pairs) == 0:
            return 0.0
            
        # Shuffle training pairs
        np.random.shuffle(training_pairs)
        
        total_loss = 0.0
        num_batches = (len(training_pairs) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(training_pairs))
            batch_pairs = training_pairs[start:end]
            actual_batch_size = len(batch_pairs)
            
            center_indices = batch_pairs[:, 0]
            context_indices = batch_pairs[:, 1]
            
            # Positive samples
            positive_labels = np.ones(actual_batch_size)
            total_loss += self._train_batch(center_indices, context_indices, positive_labels)
            
            # Negative samples
            neg_samples = self._get_negative_samples_batch(context_indices, actual_batch_size)
            
            for k in range(self.negative_samples):
                neg_indices = neg_samples[:, k]
                negative_labels = np.zeros(actual_batch_size)
                total_loss += self._train_batch(center_indices, neg_indices, negative_labels)
        
        # Calculate average loss
        total_samples = len(training_pairs) * (1 + self.negative_samples)
        return total_loss / total_samples

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

    def save_embeddings(self, filepath):
        """
        Save word embeddings to a text file in word2vec format.
        
        Args:
            filepath (str): Path to save the embeddings file.
        """
        with open(filepath, 'w') as f:
            f.write(f"{self.vocab_size} {self.embedding_dim}\n")
            for word, vec in self.word_to_vec.items():
                vec_str = ' '.join(str(v) for v in vec)
                f.write(f"{word} {vec_str}\n")
        print(f"Embeddings saved to {filepath}")

    def save_model(self, filepath):
        """
        Save the entire model as a pickle file.
        
        Args:
            filepath (str): Path to save the model file.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath):
        """
        Load a model from a pickle file.
        
        Args:
            filepath (str): Path to the saved model file.
        
        Returns:
            Word2Vec: The loaded model instance.
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model    