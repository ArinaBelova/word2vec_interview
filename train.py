"""
Training script for Word2Vec with Weights & Biases logging.
"""
import yaml
import argparse
import wandb
from word2vec import Word2Vec
from utils import load_data, preprocess_data


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
        }
    )
    
    # Load and preprocess data
    print("Loading data...")
    data = load_data(config['data']['file_path'])
    sentences = preprocess_data(data)
    print(f"Loaded {len(sentences)} sentences")
    
    # Build vocabulary stats
    vocab = set(word for sentence in sentences for word in sentence)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Log dataset info to wandb
    wandb.log({
        'num_sentences': len(sentences),
        'vocab_size': len(vocab),
    })
    
    # Initialize model
    model = Word2Vec(
        sentences=sentences,
        embedding_dim=config['model']['embedding_dim'],
        window_size=config['model']['window_size'],
        learning_rate=config['training']['learning_rate'],
        negative_samples=config['model']['negative_samples'],
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
    
    Args:
        model: Word2Vec model instance.
        epochs: Number of training epochs.
        log_interval: How often to log metrics.
    """
    import numpy as np
    
    # Generate training pairs
    training_pairs = model._generate_training_pairs()
    
    if not training_pairs:
        print("No training pairs generated. Check your data.")
        return
    
    wandb.log({'num_training_pairs': len(training_pairs)})
    print(f"Training on {len(training_pairs)} word pairs for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0.0
        np.random.shuffle(training_pairs)
        
        for center_idx, context_idx in training_pairs:
            # Positive sample
            loss = model._train_pair(center_idx, context_idx, label=1)
            total_loss += loss
            
            # Negative samples
            negative_indices = model._get_negative_samples(context_idx)
            for neg_idx in negative_indices:
                loss = model._train_pair(center_idx, neg_idx, label=0)
                total_loss += loss
        
        avg_loss = total_loss / (len(training_pairs) * (1 + model.negative_samples))
        
        # Log to wandb every epoch
        wandb.log({
            'epoch': epoch + 1,
            'loss': avg_loss,
        })
        
        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Store final embeddings
    for word, idx in model._word2idx.items():
        model.word_to_vec[word] = model.W_embed[idx].tolist()
    
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
    
    # Test the trained model
    if model.vocab:
        test_word = model.vocab[0]
        print(f"\nEmbedding for '{test_word}': {model.get_embedding(test_word)[:5]}...")
        similar = model.most_similar(test_word, top_n=3)
        print(f"Most similar to '{test_word}': {similar}")


if __name__ == "__main__":
    main()
