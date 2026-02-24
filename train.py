"""
Training script for Word2Vec with Weights & Biases logging.
"""
import yaml
import argparse
import wandb
from word2vec import Word2Vec
from utils import load_data, preprocess_data
from word2vec_datasets import (
    load_brown_corpus, 
    load_reuters_corpus, 
    load_gutenberg_corpus,
    load_combined_corpus,
    load_text8,
    load_wikitext
)
import numpy as np
from numba import set_num_threads
from tqdm import tqdm  

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_sentences(config: dict) -> list:
    """
    Load sentences based on data source configuration.
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        list: List of sentences (each sentence is a list of words).
    """
    data_config = config.get('data', {})
    source = data_config.get('source', 'file')
    max_sentences = data_config.get('max_sentences')
    
    if source == 'file':
        print(f"Loading from file: {data_config['file_path']}")
        data = load_data(data_config['file_path'])
        sentences = preprocess_data(data)
    elif source == 'brown':
        print("Loading Brown corpus...")
        sentences = load_brown_corpus()
    elif source == 'reuters':
        print("Loading Reuters corpus...")
        sentences = load_reuters_corpus()
    elif source == 'gutenberg':
        print("Loading Gutenberg corpus...")
        sentences = load_gutenberg_corpus()
    elif source == 'combined':
        print("Loading combined corpus (Brown + Reuters + Gutenberg)...")
        sentences = load_combined_corpus()
    elif source == 'text8':
        print("Loading text8 dataset...")
        sentences = load_text8(max_sentences=max_sentences)
        max_sentences = None  # Already limited in load_text8
    elif source.startswith('wikitext'):
        # Support: wikitext, wikitext-2, wikitext-103, wikitext-2-raw, wikitext-103-raw
        variant_map = {
            'wikitext': 'wikitext-103-v1',
            'wikitext-2': 'wikitext-2-v1',
            'wikitext-103': 'wikitext-103-v1',
            'wikitext-2-raw': 'wikitext-2-raw-v1',
            'wikitext-103-raw': 'wikitext-103-raw-v1',
        }
        variant = variant_map.get(source, 'wikitext-103-v1')
        print(f"Loading WikiText dataset ({variant})...")
        sentences = load_wikitext(variant=variant, max_sentences=max_sentences)
        max_sentences = None  # Already limited in load_wikitext
    else:
        raise ValueError(f"Unknown data source: {source}")
    
    # Limit sentences if specified
    if max_sentences:
        sentences = sentences[:max_sentences]
    
    return sentences


def train(config: dict):
    """
    Train Word2Vec model with wandb logging.
    
    Args:
        config: Configuration dictionary from YAML file.
    """
    # Initialize wandb
    wandb_config = config.get('wandb', {})
    wandb.init(
        project=wandb_config.get('project', 'word2vec'),
        entity=wandb_config.get('entity'),
        name=wandb_config.get('run_name'),
        tags=wandb_config.get('tags', []),
        config={
            'embedding_dim': config['model']['embedding_dim'],
            'window_size': config['model']['window_size'],
            'negative_samples': config['model']['negative_samples'],
            'learning_rate': config['training']['learning_rate'],
            'epochs': config['training']['epochs'],
            'data_source': config['data'].get('source', 'file'),
        }
    )
    
    # Load data using configured source
    sentences = load_sentences(config)
    print(f"Loaded {len(sentences)} sentences")
    
    # Build vocabulary stats
    vocab = set(word for sentence in sentences for word in sentence)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Set Numba threads if configured
    n_threads = config['model'].get('n_threads')
    if n_threads:
        set_num_threads(n_threads)
        print(f"Using {n_threads} Numba threads for parallel training")
    
    # Log dataset info to wandb
    wandb.log({
        'num_sentences': len(sentences),
        'vocab_size': len(vocab),
    })
    
    # Initialize model with optimization parameters
    model = Word2Vec(
        sentences=sentences,
        embedding_dim=config['model']['embedding_dim'],
        window_size=config['model']['window_size'],
        learning_rate=config['training']['learning_rate'],
        negative_samples=config['model']['negative_samples'],
        subsample_threshold=config['model'].get('subsample_threshold', 1e-3),
        batch_size=config['model'].get('batch_size', 4096),
    )
    
    # Train with wandb logging
    train_with_logging(
        model=model,
        epochs=config['training']['epochs'],
        log_interval=config['training'].get('log_interval', 10),
    )
    
    # Log final embeddings as artifact (optional)
    wandb.finish()
    
    return model


def train_with_logging(model: Word2Vec, epochs: int, log_interval: int = 10):
    """
    Train the model with wandb logging for loss tracking.
    Uses optimized batched training.
    
    Args:
        model: Word2Vec model instance.
        epochs: Number of training epochs.
        log_interval: How often to log metrics.
    """

    # Generate training pairs
    training_pairs = model._generate_training_pairs()
    
    if len(training_pairs) == 0:
        print("No training pairs generated. Check your data.")
        return
    
    wandb.log({'num_training_pairs': len(training_pairs)})
    print(f"Training on {len(training_pairs)} word pairs for {epochs} epochs...")
    print(f"Batch size: {model.batch_size}, Subsampling threshold: {model.subsample_threshold}")
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # Use optimized batched training
        avg_loss = model.train_epoch(training_pairs)
        
        # Log to wandb every epoch
        wandb.log({
            'epoch': epoch + 1,
            'loss': avg_loss,
        })
        
        if (epoch + 1) % log_interval == 0:
            # intermediately test the model by displaying the closest words to the choisen word and log it to wandb
            if model.vocab:
                test_word = model.vocab[np.random.randint(model.vocab_size)]
                # check if embedding for this word exists
                while model.get_embedding(test_word) is None:
                    test_word = model.vocab[np.random.randint(model.vocab_size)]
                similar = model.most_similar(test_word, top_n=3)
                print(f"Most similar to '{test_word}': {similar}")
                wandb.log({f'most_similar_to_{test_word}': similar})
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train Word2Vec model')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration YAML file'
    )
    args = parser.parse_args()
    
    # Load config and train
    config = load_config(args.config)
    model = train(config)
    
    # Save the embeddings
    model.save_embeddings(config["model"].get("save_embeddings_path", "embeddings.txt"))
    #save model as an object
    model.save_model(config["model"].get("save_model_path", "word2vec_model.pkl"))

    # Test the trained model
    if model.vocab:
        test_word = model.vocab[np.random.randint(model.vocab_size)]
        print(f"\nEmbedding for '{test_word}': {model.get_embedding(test_word)[:5]}...")
        similar = model.most_similar(test_word, top_n=3)
        print(f"Most similar to '{test_word}': {similar}")


if __name__ == "__main__":
    main()
