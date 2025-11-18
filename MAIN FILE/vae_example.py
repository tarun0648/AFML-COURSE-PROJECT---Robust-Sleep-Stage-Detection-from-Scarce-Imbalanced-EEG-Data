"""
VAE Quick Start Example
This script demonstrates how to use the VAE implementation for generating synthetic EEG data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def example_1_train_vae():
    """Example 1: Train a VAE model"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Training VAE")
    print("="*60 + "\n")
    
    from vae_trainer import VAETrainer
    
    # Configuration
    args = argparse.Namespace(
        dataset='sleep_edf',
        data_path='data',
        fold_id='0',
        data_percentage='100',
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        target_classes=[1, 3],  # N1 and N3
        balance_ratio=0.8
    )
    
    print(f"Device: {args.device}")
    print(f"Dataset: {args.dataset}")
    print(f"Fold: {args.fold_id}")
    print(f"Target classes: {args.target_classes}")
    
    # Initialize trainer
    print("\nInitializing VAE trainer...")
    trainer = VAETrainer(args)
    
    # Train
    print("\nStarting VAE training...")
    loss_history = trainer.train()
    
    print("\n✓ Training completed!")
    print(f"Final loss: {loss_history['total'][-1]:.4f}")
    
    return trainer


def example_2_generate_samples(trainer=None):
    """Example 2: Generate synthetic samples"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Generating Synthetic Samples")
    print("="*60 + "\n")
    
    if trainer is None:
        from vae_trainer import VAETrainer
        
        args = argparse.Namespace(
            dataset='sleep_edf',
            data_path='data',
            fold_id='0',
            data_percentage='100',
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            target_classes=[1, 3],
            balance_ratio=0.8
        )
        
        trainer = VAETrainer(args)
        trainer.load_model()
    
    # Compute latent statistics
    print("Computing class-specific latent statistics...")
    trainer.compute_class_latent_statistics()
    
    # Generate samples for N1 (class 1)
    print("\nGenerating 100 synthetic N1 samples...")
    samples, labels = trainer.generate_synthetic_samples(
        target_class=1,
        num_samples=100,
        output_file='example_n1_samples.pt'
    )
    
    print(f"✓ Generated {len(samples)} samples")
    print(f"  Shape: {samples.shape}")
    print(f"  Min/Max: {samples.min():.3f} / {samples.max():.3f}")
    
    # Generate samples for N3 (class 3)
    print("\nGenerating 100 synthetic N3 samples...")
    samples, labels = trainer.generate_synthetic_samples(
        target_class=3,
        num_samples=100,
        output_file='example_n3_samples.pt'
    )
    
    print(f"✓ Generated {len(samples)} samples")
    
    return trainer


def example_3_balance_dataset(trainer=None):
    """Example 3: Balance dataset with synthetic samples"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Balancing Dataset")
    print("="*60 + "\n")
    
    if trainer is None:
        from vae_trainer import VAETrainer
        
        args = argparse.Namespace(
            dataset='sleep_edf',
            data_path='data',
            fold_id='0',
            data_percentage='100',
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            target_classes=[1, 3],
            balance_ratio=0.8
        )
        
        trainer = VAETrainer(args)
        trainer.load_model()
    
    # Balance dataset
    print("Balancing dataset (ratio=0.8)...")
    synthetic_data = trainer.balance_dataset_with_synthetic(balance_ratio=0.8)
    
    print("\n✓ Dataset balanced!")
    for class_id, data in synthetic_data.items():
        print(f"  Class {class_id}: Generated {len(data['samples'])} samples")
    
    return trainer


def example_4_visualize_samples(trainer=None):
    """Example 4: Visualize real vs synthetic samples"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Visualizing Samples")
    print("="*60 + "\n")
    
    if trainer is None:
        from vae_trainer import VAETrainer
        
        args = argparse.Namespace(
            dataset='sleep_edf',
            data_path='data',
            fold_id='0',
            data_percentage='100',
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            target_classes=[1, 3],
            balance_ratio=0.8
        )
        
        trainer = VAETrainer(args)
        trainer.load_model()
        trainer.compute_class_latent_statistics()
    
    # Visualize N1 samples
    print("Creating visualization for N1 (class 1)...")
    trainer.visualize_samples(num_real=3, num_synthetic=3, target_class=1)
    
    # Visualize N3 samples
    print("Creating visualization for N3 (class 3)...")
    trainer.visualize_samples(num_real=3, num_synthetic=3, target_class=3)
    
    print("\n✓ Visualizations saved!")
    print(f"  Location: {trainer.exp_log_dir}")
    
    return trainer


def example_5_create_augmented_dataset():
    """Example 5: Create augmented training dataset"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Creating Augmented Dataset")
    print("="*60 + "\n")
    
    from augment_data_with_vae import create_single_augmented_dataset
    
    # Paths
    original_path = 'data/sleep_edf/train_0_100per.pt'
    vae_dir = 'experiments_logs_sleep_edf/VAE_Generation/Fold_0/'
    output_path = 'data/train_0_100per_augmented_example.pt'
    
    print(f"Original data: {original_path}")
    print(f"VAE data: {vae_dir}")
    print(f"Output: {output_path}")
    
    # Create augmented dataset
    print("\nMerging original and synthetic data...")
    augmented_dataset = create_single_augmented_dataset(
        original_path=original_path,
        vae_dir=vae_dir,
        output_path=output_path,
        augmentation_ratio=1.0
    )
    
    if augmented_dataset:
        print("\n✓ Augmented dataset created!")
        print(f"  Total samples: {len(augmented_dataset['samples'])}")
        print(f"  Saved to: {output_path}")
    
    return augmented_dataset


def example_6_load_and_inspect_synthetic():
    """Example 6: Load and inspect synthetic samples"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Inspecting Synthetic Samples")
    print("="*60 + "\n")
    
    # Path to synthetic data
    synthetic_file = 'experiments_logs_sleep_edf/VAE_Generation/Fold_0/synthetic_class_1_*.pt'
    
    # Find synthetic files
    from glob import glob
    files = glob(str(synthetic_file))
    
    if not files:
        print("No synthetic files found. Run generation first.")
        return None
    
    # Load first file
    file_path = files[0]
    print(f"Loading: {file_path}")
    
    data = torch.load(file_path)
    samples = data['samples']
    labels = data['labels']
    
    print(f"\n✓ Loaded synthetic data:")
    print(f"  Samples shape: {samples.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique labels: {np.unique(labels)}")
    print(f"  Sample statistics:")
    print(f"    Mean: {samples.mean():.4f}")
    print(f"    Std: {samples.std():.4f}")
    print(f"    Min: {samples.min():.4f}")
    print(f"    Max: {samples.max():.4f}")
    
    # Plot a sample
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(samples[0, 0, :])
    ax.set_title('Example Synthetic EEG Sample')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    
    output_file = 'example_synthetic_sample.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Plot saved to: {output_file}")
    
    return data


def main():
    parser = argparse.ArgumentParser(description='VAE Examples')
    parser.add_argument('--example', type=int, default=0, 
                       help='Example number to run (1-6, or 0 for all)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (skip training)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("VAE for EEG Data Generation - Examples")
    print("="*60)
    
    if args.example == 0:
        print("\nRunning all examples...")
        print("Note: This will take some time!")
        
        # Check if we should skip training
        if args.quick:
            print("\n[Quick mode: Skipping training]")
            trainer = None
        else:
            # Example 1: Train
            trainer = example_1_train_vae()
        
        # Example 2: Generate
        trainer = example_2_generate_samples(trainer)
        
        # Example 3: Balance
        trainer = example_3_balance_dataset(trainer)
        
        # Example 4: Visualize
        trainer = example_4_visualize_samples(trainer)
        
        # Example 5: Create augmented dataset
        example_5_create_augmented_dataset()
        
        # Example 6: Inspect
        example_6_load_and_inspect_synthetic()
        
    elif args.example == 1:
        example_1_train_vae()
    elif args.example == 2:
        example_2_generate_samples()
    elif args.example == 3:
        example_3_balance_dataset()
    elif args.example == 4:
        example_4_visualize_samples()
    elif args.example == 5:
        example_5_create_augmented_dataset()
    elif args.example == 6:
        example_6_load_and_inspect_synthetic()
    else:
        print(f"Invalid example number: {args.example}")
        return
    
    print("\n" + "="*60)
    print("Examples Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

