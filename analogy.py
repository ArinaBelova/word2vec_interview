"""
Word analogy experiments using saved Word2Vec embeddings.
Tests relationships like: king - man + woman ≈ queen
"""
import numpy as np
import argparse


def load_embeddings(filepath):
    """
    Load embeddings from word2vec text format.
    
    Args:
        filepath: Path to embeddings file.
    
    Returns:
        word_to_vec: Dictionary mapping words to numpy vectors.
        vocab: List of words in vocabulary.
    """
    word_to_vec = {}
    with open(filepath, 'r') as f:
        # First line: vocab_size embedding_dim
        header = f.readline().strip().split()
        vocab_size, embedding_dim = int(header[0]), int(header[1])
        print(f"Loading {vocab_size} embeddings of dimension {embedding_dim}")
        
        for line in f:
            parts = line.strip().split()
            if len(parts) < embedding_dim + 1:
                continue
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            word_to_vec[word] = vec
    
    vocab = list(word_to_vec.keys())
    print(f"Loaded {len(vocab)} words")
    return word_to_vec, vocab


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def find_most_similar(target_vec, word_to_vec, exclude_words=None, top_n=5):
    """
    Find words most similar to a target vector.
    
    Args:
        target_vec: Target embedding vector.
        word_to_vec: Dictionary of word embeddings.
        exclude_words: Set of words to exclude from results.
        top_n: Number of results to return.
    
    Returns:
        List of (word, similarity) tuples.
    """
    if exclude_words is None:
        exclude_words = set()
    
    similarities = []
    for word, vec in word_to_vec.items():
        if word not in exclude_words:
            sim = cosine_similarity(target_vec, vec)
            similarities.append((word, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


def analogy(word_a, word_b, word_c, word_to_vec, top_n=5):
    """
    Solve analogy: a is to b as c is to ?
    Computes: vec(b) - vec(a) + vec(c)
    
    Example: king is to man as queen is to woman
             king - man + woman ≈ queen
    
    Args:
        word_a: First word (e.g., "man")
        word_b: Second word (e.g., "king") 
        word_c: Third word (e.g., "woman")
        word_to_vec: Dictionary of word embeddings.
        top_n: Number of results to return.
    
    Returns:
        List of (word, similarity) tuples, or None if words not found.
    """
    # Check all words exist
    for word in [word_a, word_b, word_c]:
        if word not in word_to_vec:
            print(f"Word '{word}' not in vocabulary")
            return None
    
    # Compute analogy vector: b - a + c
    # "king - man + woman = queen" means king is to man as queen is to woman
    vec_a = word_to_vec[word_a]
    vec_b = word_to_vec[word_b]
    vec_c = word_to_vec[word_c]
    
    target_vec = vec_b - vec_a + vec_c
    
    # Find most similar, excluding input words
    exclude = {word_a, word_b, word_c}
    return find_most_similar(target_vec, word_to_vec, exclude, top_n)


def run_analogy_tests(word_to_vec):
    """Run a set of common analogy tests."""
    
    # Common analogy test cases
    # Format: (a, b, c, expected_d) where b - a + c ≈ d
    test_cases = [
        # Gender analogies
        ("man", "king", "woman", "queen"),
        ("man", "brother", "woman", "sister"),
        ("man", "he", "woman", "she"),
        ("boy", "man", "girl", "woman"),
        
        # Verb tense
        ("walk", "walked", "run", "ran"),
        ("go", "went", "come", "came"),
        
        # Country-capital
        ("france", "paris", "germany", "berlin"),
        ("japan", "tokyo", "china", "beijing"),
        
        # Comparative/superlative
        ("good", "better", "bad", "worse"),
        ("big", "bigger", "small", "smaller"),
        
        # Plural
        ("car", "cars", "dog", "dogs"),
        ("child", "children", "man", "men"),
    ]
    
    print("\n" + "=" * 60)
    print("WORD ANALOGY TESTS")
    print("Format: a is to b as c is to ? (expected: d)")
    print("=" * 60 + "\n")
    
    correct = 0
    tested = 0
    
    for word_a, word_b, word_c, expected in test_cases:
        results = analogy(word_a, word_b, word_c, word_to_vec, top_n=5)
        
        if results is None:
            print(f"SKIP: {word_a}:{word_b}::{word_c}:? - missing words")
            continue
        
        tested += 1
        top_word = results[0][0]
        is_correct = top_word == expected
        
        if is_correct:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{status} {word_a}:{word_b}::{word_c}:? (expected: {expected})")
        print(f"  Top 5: {[(w, f'{s:.3f}') for w, s in results]}")
        print()
    
    if tested > 0:
        accuracy = correct / tested * 100
        print(f"\nAccuracy: {correct}/{tested} ({accuracy:.1f}%)")
    else:
        print("\nNo tests could be run - vocabulary missing required words")


def interactive_mode(word_to_vec):
    """Interactive mode for custom analogy queries."""
    print("\n" + "=" * 60)
    print("INTERACTIVE ANALOGY MODE")
    print("Enter three words: a b c")
    print("Computes: b - a + c = ?")
    print("Example: 'man king woman' finds words like 'queen'")
    print("Type 'quit' to exit")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("Enter words (a b c): ").strip().lower()
        except EOFError:
            break
            
        if user_input == 'quit' or user_input == 'q':
            break
        
        parts = user_input.split()
        if len(parts) != 3:
            print("Please enter exactly 3 words")
            continue
        
        word_a, word_b, word_c = parts
        results = analogy(word_a, word_b, word_c, word_to_vec, top_n=10)
        
        if results:
            print(f"\n{word_a}:{word_b}::{word_c}:?")
            print("Results:")
            for i, (word, sim) in enumerate(results, 1):
                print(f"  {i}. {word} ({sim:.4f})")
            print()


def main():
    parser = argparse.ArgumentParser(description='Word analogy experiments')
    parser.add_argument(
        '--embeddings',
        type=str,
        default='embeddings.txt',
        help='Path to saved embeddings file'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    args = parser.parse_args()
    
    # Load embeddings
    word_to_vec, vocab = load_embeddings(args.embeddings)
    
    # Run standard tests
    run_analogy_tests(word_to_vec)
    
    # Interactive mode if requested
    if args.interactive:
        interactive_mode(word_to_vec)


if __name__ == "__main__":
    main()
