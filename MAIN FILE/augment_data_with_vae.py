"""
Data Augmentation Utility using VAE-generated Synthetic Samples
This script merges synthetic samples with original training data to create balanced datasets.
"""

import torch
import os
import argparse
import numpy as np
from pathlib import Path


def load_synthetic_data(vae_experiment_dir):
    """Load all synthetic data files from VAE experiment directory"""
    synthetic_files = list(Path(vae_experiment_dir).glob('synthetic_class_*.pt'))
    
    synthetic_data = {}
    for file in synthetic_files:
        data = torch.load(file, weights_only=False)
        # Extract class number from filename
        class_id = int(file.stem.split('_')[2])
        synthetic_data[class_id] = data
        print(f"Loaded {len(data['samples'])} synthetic samples for class {class_id} from {file.name}")
    
    return synthetic_data


def merge_with_original(original_data_path, synthetic_data, output_path, augmentation_ratio=1.0):
    """
    Merge synthetic samples with original training data
    
    Args:
        original_data_path: Path to original training data file
        synthetic_data: Dictionary of synthetic data by class
        output_path: Path to save augmented dataset
        augmentation_ratio: Ratio of synthetic samples to add (1.0 = add all generated samples)
    """
    # Load original data
    original = torch.load(original_data_path, weights_only=False)
    original_samples = original['samples']
    original_labels = original['labels']
    
    # Convert to numpy if needed
    if isinstance(original_samples, torch.Tensor):
        original_samples = original_samples.numpy()
    if isinstance(original_labels, torch.Tensor):
        original_labels = original_labels.numpy()
    
    print(f"\nOriginal dataset: {len(original_samples)} samples")
    
    # Analyze original class distribution
    unique, counts = np.unique(original_labels, return_counts=True)
    print("\nOriginal class distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples ({count/len(original_labels)*100:.2f}%)")
    
    # Merge with synthetic data
    all_samples = [original_samples]
    all_labels = [original_labels]
    
    total_synthetic = 0
    for class_id, synth_data in synthetic_data.items():
        synth_samples = synth_data['samples']
        synth_labels = synth_data['labels']
        
        # Apply augmentation ratio
        num_to_add = int(len(synth_samples) * augmentation_ratio)
        if num_to_add > 0:
            all_samples.append(synth_samples[:num_to_add])
            all_labels.append(synth_labels[:num_to_add])
            total_synthetic += num_to_add
            print(f"  Adding {num_to_add} synthetic samples for class {class_id}")
    
    # Concatenate all data
    augmented_samples = np.concatenate(all_samples, axis=0)
    augmented_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\nAugmented dataset: {len(augmented_samples)} samples ({total_synthetic} synthetic)")
    
    # Analyze augmented class distribution
    unique, counts = np.unique(augmented_labels, return_counts=True)
    print("\nAugmented class distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples ({count/len(augmented_labels)*100:.2f}%)")
    
    # Shuffle the dataset
    shuffle_idx = np.random.permutation(len(augmented_samples))
    augmented_samples = augmented_samples[shuffle_idx]
    augmented_labels = augmented_labels[shuffle_idx]
    
    # Save augmented dataset
    augmented_dataset = {
        'samples': augmented_samples,
        'labels': augmented_labels
    }
    
    torch.save(augmented_dataset, output_path)
    print(f"\nSaved augmented dataset to {output_path}")
    
    return augmented_dataset


def create_augmented_datasets_for_all_folds(
    data_dir, 
    vae_experiments_dir, 
    dataset_name='sleep_edf',
    data_percentages=['1', '5', '10', '50', '75', '100'],
    num_folds=5,
    augmentation_ratio=1.0
):
    """
    Create augmented datasets for all folds and data percentages
    
    Args:
        data_dir: Directory containing original data
        vae_experiments_dir: Directory containing VAE experiments
        dataset_name: Name of the dataset
        data_percentages: List of data percentages to process
        num_folds: Number of folds
        augmentation_ratio: Ratio of synthetic samples to add
    """
    print("=" * 60)
    print("Creating Augmented Datasets for All Folds")
    print("=" * 60)
    
    # Create output directory
    augmented_data_dir = os.path.join(data_dir, f'{dataset_name}_augmented')
    os.makedirs(augmented_data_dir, exist_ok=True)
    
    for fold_id in range(num_folds):
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold_id}")
        print(f"{'='*60}")
        
        # Load synthetic data for this fold
        vae_fold_dir = os.path.join(vae_experiments_dir, 'VAE_Generation', f'Fold_{fold_id}')
        
        if not os.path.exists(vae_fold_dir):
            print(f"Warning: VAE experiment directory not found for fold {fold_id}")
            print(f"Expected: {vae_fold_dir}")
            print("Skipping this fold...")
            continue
        
        try:
            synthetic_data = load_synthetic_data(vae_fold_dir)
            
            if len(synthetic_data) == 0:
                print(f"No synthetic data found for fold {fold_id}, skipping...")
                continue
            
            # Process each data percentage
            for data_pct in data_percentages:
                original_file = os.path.join(data_dir, dataset_name, f'train_{fold_id}_{data_pct}per.pt')
                
                if not os.path.exists(original_file):
                    print(f"Warning: Original file not found: {original_file}")
                    continue
                
                output_file = os.path.join(augmented_data_dir, f'train_{fold_id}_{data_pct}per_augmented.pt')
                
                print(f"\n--- Processing {data_pct}% data ---")
                merge_with_original(
                    original_file, 
                    synthetic_data, 
                    output_file,
                    augmentation_ratio=augmentation_ratio
                )
        
        except Exception as e:
            print(f"Error processing fold {fold_id}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("Augmented dataset creation complete!")
    print(f"Augmented datasets saved to: {augmented_data_dir}")
    print("=" * 60)


def create_single_augmented_dataset(original_path, vae_dir, output_path, augmentation_ratio=1.0):
    """Create a single augmented dataset"""
    print(f"\nCreating augmented dataset from:")
    print(f"  Original: {original_path}")
    print(f"  VAE data: {vae_dir}")
    print(f"  Output: {output_path}")
    print(f"  Augmentation ratio: {augmentation_ratio}")
    
    # Load synthetic data
    synthetic_data = load_synthetic_data(vae_dir)
    
    if len(synthetic_data) == 0:
        print("No synthetic data found!")
        return None
    
    # Merge and save
    augmented_dataset = merge_with_original(
        original_path,
        synthetic_data,
        output_path,
        augmentation_ratio=augmentation_ratio
    )
    
    return augmented_dataset


def main():
    parser = argparse.ArgumentParser(description='Augment Training Data with VAE-generated Samples')
    
    parser.add_argument('--mode', default='all_folds', type=str, 
                       choices=['single', 'all_folds'],
                       help='Mode: single file or all folds')
    
    # For single mode
    parser.add_argument('--original_path', type=str,
                       help='Path to original training data file')
    parser.add_argument('--vae_dir', type=str,
                       help='Directory containing VAE-generated synthetic data')
    parser.add_argument('--output_path', type=str,
                       help='Path to save augmented dataset')
    
    # For all_folds mode
    parser.add_argument('--data_dir', default='data', type=str,
                       help='Root directory containing all data')
    parser.add_argument('--vae_experiments_dir', type=str,
                       help='Directory containing VAE experiments (e.g., experiments_logs_sleep_edf)')
    parser.add_argument('--dataset_name', default='sleep_edf', type=str,
                       help='Dataset name')
    parser.add_argument('--num_folds', default=5, type=int,
                       help='Number of folds')
    parser.add_argument('--data_percentages', nargs='+', 
                       default=['1', '5', '10', '50', '75', '100'],
                       help='List of data percentages to process')
    
    # Common parameters
    parser.add_argument('--augmentation_ratio', default=1.0, type=float,
                       help='Ratio of synthetic samples to add (0.0-1.0)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not all([args.original_path, args.vae_dir, args.output_path]):
            parser.error("For single mode, --original_path, --vae_dir, and --output_path are required")
        
        create_single_augmented_dataset(
            args.original_path,
            args.vae_dir,
            args.output_path,
            args.augmentation_ratio
        )
    
    elif args.mode == 'all_folds':
        if not args.vae_experiments_dir:
            # Try to infer from data_dir and dataset_name
            args.vae_experiments_dir = f'experiments_logs_{args.dataset_name}'
        
        create_augmented_datasets_for_all_folds(
            args.data_dir,
            args.vae_experiments_dir,
            args.dataset_name,
            args.data_percentages,
            args.num_folds,
            args.augmentation_ratio
        )


if __name__ == "__main__":
    main()

