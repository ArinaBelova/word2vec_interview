"""
Comprehensive test suite for Word2Vec implementation.
Tests preprocessing, model initialization, training, and embeddings.
"""
import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from word2vec import Word2Vec
from utils import load_data, preprocess_data
from datasets import load_brown_corpus


class TestPreprocessing:
    """Tests for data preprocessing functions."""
    
    def test_load_data_returns_list(self, tmp_path):
        """Test that load_data returns a list of lines."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world.\nThis is a test.")
        
        data = load_data(str(test_file))
        assert isinstance(data, list)
        assert len(data) == 2
    
    def test_preprocess_removes_stopwords(self):
        """Test that preprocessing removes stopwords."""
        data = ["The quick brown fox jumps over the lazy dog."]
        result = preprocess_data(data)
        
        # 'the' and 'over' should be removed
        assert len(result) == 1
        words = result[0]
        assert 'the' not in words
        assert 'over' not in words
        assert 'quick' in words
        assert 'fox' in words
    
    def test_preprocess_lowercases_words(self):
        """Test that preprocessing lowercases all words."""
        data = ["The QUICK Brown FOX"]
        result = preprocess_data(data)
        
        words = result[0]
        for word in words:
            assert word == word.lower()
    
    def test_preprocess_removes_punctuation(self):
        """Test that preprocessing removes punctuation."""
        data = ["Hello, world! How are you?"]
        result = preprocess_data(data)
        
        words = result[0]
        for word in words:
            assert word.isalpha()
    
    def test_preprocess_keeps_sentence_structure(self):
        """Test that sentences are kept separate."""
        data = ["First sentence here.", "Second sentence here."]
        result = preprocess_data(data)
        
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)
    
    def test_preprocess_empty_input(self):
        """Test preprocessing with empty input."""
        result = preprocess_data([])
        assert result == []
    
    def test_preprocess_stopword_only_sentence(self):
        """Test that sentences with only stopwords are filtered out."""
        data = ["The a an the"]
        result = preprocess_data(data)
        # Should be empty since all words are stopwords
        assert len(result) == 0


class TestWord2VecInitialization:
    """Tests for Word2Vec model initialization."""
    
    @pytest.fixture
    def sample_sentences(self):
        return [
            ['quick', 'brown', 'fox', 'jumps'],
            ['lazy', 'dog', 'sleeps'],
            ['cat', 'sits', 'mat']
        ]
    
    def test_vocabulary_building(self, sample_sentences):
        """Test that vocabulary is built correctly."""
        model = Word2Vec(sample_sentences, embedding_dim=50)
        
        expected_vocab = {'quick', 'brown', 'fox', 'jumps', 'lazy', 'dog', 'sleeps', 'cat', 'sits', 'mat'}
        assert set(model.vocab) == expected_vocab
        assert model.vocab_size == 10
    
    def test_word_to_index_mapping(self, sample_sentences):
        """Test word2idx creates correct mappings."""
        model = Word2Vec(sample_sentences, embedding_dim=50)
        
        word2idx = model.word2idx()
        assert len(word2idx) == model.vocab_size
        
        # Each word should have a unique index
        indices = list(word2idx.values())
        assert len(set(indices)) == len(indices)
        
        # Indices should be 0 to vocab_size-1
        assert min(indices) == 0
        assert max(indices) == model.vocab_size - 1
    
    def test_index_to_word_mapping(self, sample_sentences):
        """Test idx2word creates correct inverse mappings."""
        model = Word2Vec(sample_sentences, embedding_dim=50)
        
        word2idx = model.word2idx()
        idx2word = model.idx2word()
        
        for word, idx in word2idx.items():
            assert idx2word[idx] == word
    
    def test_embedding_matrix_shape(self, sample_sentences):
        """Test embedding matrices have correct shape."""
        embedding_dim = 50
        model = Word2Vec(sample_sentences, embedding_dim=embedding_dim)
        
        assert model.W_embed.shape == (model.vocab_size, embedding_dim)
        assert model.W_context.shape == (model.vocab_size, embedding_dim)
    
    def test_embedding_initialization(self, sample_sentences):
        """Test embeddings are initialized with small random values."""
        model = Word2Vec(sample_sentences, embedding_dim=50)
        
        # Should be small values (initialized with * 0.01)
        assert np.abs(model.W_embed).max() < 0.1
        assert np.abs(model.W_context).max() < 0.1
    
    def test_hyperparameters_stored(self, sample_sentences):
        """Test that hyperparameters are stored correctly."""
        model = Word2Vec(
            sample_sentences, 
            embedding_dim=100, 
            window_size=3, 
            learning_rate=0.05,
            negative_samples=10
        )
        
        assert model.embedding_dim == 100
        assert model.window_size == 3
        assert model.learning_rate == 0.05
        assert model.negative_samples == 10


class TestTrainingPairs:
    """Tests for skip-gram training pair generation."""
    
    def test_generate_training_pairs_basic(self):
        """Test training pair generation for simple case."""
        sentences = [['a', 'b', 'c']]
        model = Word2Vec(sentences, embedding_dim=10, window_size=1)
        
        pairs = model._generate_training_pairs()
        
        # With window=1: (a,b), (b,a), (b,c), (c,b) = 4 pairs
        assert len(pairs) == 4
    
    def test_training_pairs_respect_window(self):
        """Test that training pairs respect window size."""
        sentences = [['a', 'b', 'c', 'd', 'e']]
        model = Word2Vec(sentences, embedding_dim=10, window_size=1)
        
        pairs = model._generate_training_pairs()
        
        # Convert to readable format
        idx2word = model.idx2word()
        readable_pairs = [(idx2word[c], idx2word[ctx]) for c, ctx in pairs]
        
        # 'a' should only pair with 'b' (not 'c', 'd', 'e')
        a_pairs = [p for p in readable_pairs if p[0] == 'a']
        assert all(p[1] == 'b' for p in a_pairs)
    
    def test_training_pairs_respect_sentence_boundaries(self):
        """Test that training pairs don't cross sentence boundaries."""
        sentences = [['a', 'b'], ['c', 'd']]
        model = Word2Vec(sentences, embedding_dim=10, window_size=2)
        
        pairs = model._generate_training_pairs()
        idx2word = model.idx2word()
        readable_pairs = [(idx2word[c], idx2word[ctx]) for c, ctx in pairs]
        
        # 'b' should not pair with 'c' or 'd'
        b_pairs = [p for p in readable_pairs if p[0] == 'b']
        for _, ctx in b_pairs:
            assert ctx in ['a']  # Only 'a' is in same sentence
    
    def test_no_self_pairs(self):
        """Test that a word is not paired with itself."""
        sentences = [['a', 'b', 'c']]
        model = Word2Vec(sentences, embedding_dim=10, window_size=2)
        
        pairs = model._generate_training_pairs()
        
        for center, context in pairs:
            assert center != context


class TestTraining:
    """Tests for model training."""
    
    @pytest.fixture
    def trained_model(self):
        """Create and train a model on sample data."""
        sentences = [
            ['king', 'queen', 'prince', 'princess'],
            ['man', 'woman', 'boy', 'girl'],
            ['king', 'man', 'throne', 'crown'],
            ['queen', 'woman', 'throne', 'crown'],
            ['prince', 'boy', 'young', 'heir'],
            ['princess', 'girl', 'young', 'heir'],
        ]
        model = Word2Vec(
            sentences, 
            embedding_dim=30, 
            window_size=2,
            learning_rate=0.1,
            negative_samples=3
        )
        
        # Train for a few epochs
        training_pairs = model._generate_training_pairs()
        for _ in range(50):
            np.random.shuffle(training_pairs)
            for center_idx, context_idx in training_pairs:
                model._train_pair(center_idx, context_idx, label=1)
                neg_indices = model._get_negative_samples(context_idx)
                for neg_idx in neg_indices:
                    model._train_pair(center_idx, neg_idx, label=0)
        
        # Store embeddings
        for word, idx in model._word2idx.items():
            model.word_to_vec[word] = model.W_embed[idx].tolist()
        
        return model
    
    def test_loss_decreases(self):
        """Test that loss decreases during training."""
        sentences = [['a', 'b', 'c', 'd', 'e']] * 10
        model = Word2Vec(sentences, embedding_dim=20, window_size=2, learning_rate=0.1)
        
        training_pairs = model._generate_training_pairs()
        
        # Calculate initial loss
        initial_loss = 0
        for center_idx, context_idx in training_pairs[:20]:
            initial_loss += model._train_pair(center_idx, context_idx, label=1)
        
        # Reinitialize and train
        model.W_embed = np.random.randn(model.vocab_size, model.embedding_dim) * 0.01
        model.W_context = np.random.randn(model.vocab_size, model.embedding_dim) * 0.01
        
        for _ in range(100):
            for center_idx, context_idx in training_pairs:
                model._train_pair(center_idx, context_idx, label=1)
        
        # Calculate final loss
        final_loss = 0
        for center_idx, context_idx in training_pairs[:20]:
            center_vec = model.W_embed[center_idx]
            context_vec = model.W_context[context_idx]
            dot = np.dot(center_vec, context_vec)
            pred = 1 / (1 + np.exp(-np.clip(dot, -500, 500)))
            final_loss += -np.log(pred + 1e-7)
        
        assert final_loss < initial_loss
    
    def test_negative_sampling(self):
        """Test negative sampling generates correct number of samples."""
        sentences = [['a', 'b', 'c', 'd', 'e', 'f', 'g']]
        model = Word2Vec(sentences, embedding_dim=10, negative_samples=5)
        
        neg_samples = model._get_negative_samples(0)
        
        assert len(neg_samples) == 5
        assert 0 not in neg_samples  # Should not include positive index
    
    def test_sigmoid_numerical_stability(self):
        """Test sigmoid handles extreme values without overflow."""
        sentences = [['a', 'b']]
        model = Word2Vec(sentences, embedding_dim=10)
        
        # Test extreme positive and negative values
        result_pos = model._sigmoid(1000)
        result_neg = model._sigmoid(-1000)
        
        assert np.isfinite(result_pos)
        assert np.isfinite(result_neg)
        assert 0 < result_pos <= 1
        assert 0 <= result_neg < 1
    
    def test_embeddings_stored_after_training(self, trained_model):
        """Test that embeddings are stored in word_to_vec after training."""
        assert len(trained_model.word_to_vec) == trained_model.vocab_size
        
        for word in trained_model.vocab:
            assert word in trained_model.word_to_vec
            assert len(trained_model.word_to_vec[word]) == trained_model.embedding_dim


class TestEmbeddings:
    """Tests for embedding retrieval and similarity."""
    
    @pytest.fixture
    def trained_model(self):
        """Create a trained model for embedding tests."""
        sentences = [
            ['dog', 'cat', 'pet', 'animal'],
            ['car', 'truck', 'vehicle', 'drive'],
            ['dog', 'bark', 'pet'],
            ['cat', 'meow', 'pet'],
            ['car', 'drive', 'road'],
            ['truck', 'drive', 'road'],
        ] * 20  # Repeat for more training data
        
        model = Word2Vec(
            sentences, 
            embedding_dim=50, 
            window_size=2,
            learning_rate=0.1,
            negative_samples=5
        )
        
        # Train
        training_pairs = model._generate_training_pairs()
        for _ in range(100):
            np.random.shuffle(training_pairs)
            for center_idx, context_idx in training_pairs:
                model._train_pair(center_idx, context_idx, label=1)
                for neg_idx in model._get_negative_samples(context_idx):
                    model._train_pair(center_idx, neg_idx, label=0)
        
        # Store embeddings
        for word, idx in model._word2idx.items():
            model.word_to_vec[word] = model.W_embed[idx].tolist()
        
        return model
    
    def test_get_embedding_known_word(self, trained_model):
        """Test getting embedding for a known word."""
        embedding = trained_model.get_embedding('dog')
        
        assert len(embedding) == trained_model.embedding_dim
        assert not all(v == 0 for v in embedding)
    
    def test_get_embedding_unknown_word(self, trained_model):
        """Test getting embedding for unknown word returns zero vector."""
        embedding = trained_model.get_embedding('xyznotaword')
        
        assert len(embedding) == trained_model.embedding_dim
        assert all(v == 0 for v in embedding)
    
    def test_most_similar_returns_correct_count(self, trained_model):
        """Test most_similar returns requested number of results."""
        similar = trained_model.most_similar('dog', top_n=3)
        
        assert len(similar) <= 3
    
    def test_most_similar_excludes_query_word(self, trained_model):
        """Test most_similar doesn't include the query word itself."""
        similar = trained_model.most_similar('dog', top_n=10)
        
        words = [word for word, _ in similar]
        assert 'dog' not in words
    
    def test_most_similar_returns_similarity_scores(self, trained_model):
        """Test that similarity scores are in valid range [-1, 1]."""
        similar = trained_model.most_similar('dog', top_n=5)
        
        for word, score in similar:
            assert -1 <= score <= 1
    
    def test_most_similar_sorted_descending(self, trained_model):
        """Test that results are sorted by similarity (descending)."""
        similar = trained_model.most_similar('dog', top_n=5)
        
        scores = [score for _, score in similar]
        assert scores == sorted(scores, reverse=True)
    
    def test_most_similar_unknown_word(self, trained_model):
        """Test most_similar with unknown word returns empty list."""
        similar = trained_model.most_similar('xyznotaword')
        
        assert similar == []


class TestIntegration:
    """Integration tests using real corpus data."""
    
    def test_train_on_brown_corpus_subset(self):
        """Test training on a subset of Brown corpus."""
        sentences = load_brown_corpus()[:1000]  # First 1000 sentences
        
        model = Word2Vec(
            sentences,
            embedding_dim=50,
            window_size=2,
            learning_rate=0.05,
            negative_samples=5
        )
        
        assert model.vocab_size > 100  # Should have significant vocabulary
        
        # Train briefly
        training_pairs = model._generate_training_pairs()
        for _ in range(10):
            for center_idx, context_idx in training_pairs[:1000]:
                model._train_pair(center_idx, context_idx, label=1)
        
        # Store embeddings
        for word, idx in model._word2idx.items():
            model.word_to_vec[word] = model.W_embed[idx].tolist()
        
        # Check we can get embeddings
        if 'the' not in model._word2idx:
            test_word = list(model._word2idx.keys())[0]
        else:
            test_word = list(model._word2idx.keys())[0]
        
        embedding = model.get_embedding(test_word)
        assert len(embedding) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
