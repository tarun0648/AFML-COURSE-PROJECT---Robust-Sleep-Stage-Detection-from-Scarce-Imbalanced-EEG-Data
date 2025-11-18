"""
VAE Trainer for Generating Synthetic EEG Data
This script trains a VAE on EEG data and generates synthetic samples for minority classes.
"""

import torch
import torch.nn as nn
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from models.models import VAE_EEG, vae_loss_function
from algorithms import vae
from configs.data_configs import get_dataset_class
from configs.hparams import get_hparams_class
from utils import fix_randomness, to_device, starting_logs, save_checkpoint
from dataloader.dataloader import Load_Dataset
import matplotlib.pyplot as plt


class VAETrainer:
    """Trainer class for VAE model"""
    
    def __init__(self, args):
        self.dataset = args.dataset
        self.fold_id = args.fold_id
        self.data_percentage = args.data_percentage
        self.device = torch.device(args.device)
        self.target_classes = args.target_classes  # Classes to generate (e.g., [1, 3] for N1 and N3)
        
        # Paths
        self.home_path = os.getcwd()
        self.data_path = args.data_path
        self.save_dir = f'experiments_logs_{self.dataset}'
        self.exp_log_dir = os.path.join(self.save_dir, 'VAE_Generation', f'Fold_{self.fold_id}')
        os.makedirs(self.exp_log_dir, exist_ok=True)
        
        # Get configs
        self.dataset_configs = get_dataset_class(self.dataset)()
        self.hparams_class = get_hparams_class('vae')()
        self.hparams = {**self.hparams_class.train_params}
        
        if self.dataset == "sleep_edf":
            self.hparams.update(self.hparams_class.alg_hparams_edf)
        elif self.dataset == "shhs":
            self.hparams.update(self.hparams_class.alg_hparams_shhs)
        elif self.dataset == "isruc":
            self.hparams.update(self.hparams_class.alg_hparams_isruc)
        
        # Fix randomness
        fix_randomness(int(self.fold_id))
        
        # Load data
        self.load_data()
        
        # Initialize model
        self.vae_model = VAE_EEG(self.dataset_configs, latent_dim=self.hparams['latent_dim'])
        self.vae_model.to(self.device)
        
        # Initialize algorithm
        self.algorithm = vae(self.vae_model, self.dataset_configs, self.hparams, self.device)
        
        print(f"VAE initialized with {sum(p.numel() for p in self.vae_model.parameters())} parameters")
        
    def load_data(self):
        """Load training data"""
        train_dataset = torch.load(
            os.path.join(self.data_path, self.dataset, 
                        f"train_{self.fold_id}_{self.data_percentage}per.pt")
        )
        
        # Create dataset
        train_dataset = Load_Dataset(
            train_dataset, 
            self.dataset_configs, 
            train_mode='supervised',  # Use supervised mode to get all samples
            ssl_method='supervised', 
            augmentation='',
            oversample=False
        )
        
        # Create dataloader
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, 
            batch_size=self.hparams["batch_size"],
            shuffle=True, 
            drop_last=True, 
            num_workers=0
        )
        
        print(f"Loaded {len(train_dataset)} training samples")
        
        # Analyze class distribution
        self.analyze_class_distribution(train_dataset)
        
    def analyze_class_distribution(self, dataset):
        """Analyze and print class distribution"""
        labels = dataset.y_data.numpy()
        unique, counts = np.unique(labels, return_counts=True)
        
        print("\n=== Class Distribution ===")
        for cls, count in zip(unique, counts):
            class_name = self.dataset_configs.class_names[cls]
            print(f"{class_name} (Class {cls}): {count} samples ({count/len(labels)*100:.2f}%)")
        print("=" * 30 + "\n")
        
        self.class_counts = dict(zip(unique, counts))
        
    def train(self):
        """Train the VAE model"""
        print("Starting VAE training...")
        
        best_loss = float('inf')
        loss_history = {'total': [], 'recon': [], 'kl': []}
        
        for epoch in range(1, self.hparams['num_epochs'] + 1):
            self.vae_model.train()
            epoch_losses = {'total': [], 'recon': [], 'kl': []}
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.hparams["num_epochs"]}')
            for data in pbar:
                data = to_device(data, self.device)
                
                # Update VAE
                losses, _ = self.algorithm.update(data)
                
                epoch_losses['total'].append(losses['Total_loss'])
                epoch_losses['recon'].append(losses['Recon_loss'])
                epoch_losses['kl'].append(losses['KL_loss'])
                
                pbar.set_postfix({
                    'Total': f"{losses['Total_loss']:.4f}",
                    'Recon': f"{losses['Recon_loss']:.4f}",
                    'KL': f"{losses['KL_loss']:.4f}"
                })
            
            # Calculate epoch averages
            avg_total = np.mean(epoch_losses['total'])
            avg_recon = np.mean(epoch_losses['recon'])
            avg_kl = np.mean(epoch_losses['kl'])
            
            loss_history['total'].append(avg_total)
            loss_history['recon'].append(avg_recon)
            loss_history['kl'].append(avg_kl)
            
            print(f"Epoch {epoch}: Total Loss = {avg_total:.4f}, Recon = {avg_recon:.4f}, KL = {avg_kl:.4f}")
            
            # Save best model
            if avg_total < best_loss:
                best_loss = avg_total
                self.save_model()
                print(f"Saved best model with loss {best_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_model(filename=f'checkpoint_epoch_{epoch}.pt')
        
        # Plot training curves
        self.plot_training_curves(loss_history)
        
        print("\nTraining completed!")
        return loss_history
    
    def save_model(self, filename='vae_best_model.pt'):
        """Save the trained VAE model"""
        save_path = os.path.join(self.exp_log_dir, filename)
        torch.save({
            'vae_state_dict': self.vae_model.state_dict(),
            'optimizer_state_dict': self.algorithm.optimizer.state_dict(),
            'hparams': self.hparams,
            'dataset_configs': self.dataset_configs
        }, save_path)
        
    def load_model(self, filename='vae_best_model.pt'):
        """Load a trained VAE model"""
        load_path = os.path.join(self.exp_log_dir, filename)
        checkpoint = torch.load(load_path, map_location=self.device)
        self.vae_model.load_state_dict(checkpoint['vae_state_dict'])
        print(f"Loaded model from {load_path}")
        
    def plot_training_curves(self, loss_history):
        """Plot training loss curves"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(loss_history['total'])
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        
        axes[1].plot(loss_history['recon'])
        axes[1].set_title('Reconstruction Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True)
        
        axes[2].plot(loss_history['kl'])
        axes[2].set_title('KL Divergence')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_log_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def compute_class_latent_statistics(self):
        """Compute mean and std of latent vectors for each class"""
        print("\nComputing class-specific latent statistics...")
        
        self.vae_model.eval()
        class_latents = {i: [] for i in range(self.dataset_configs.num_classes)}
        
        with torch.no_grad():
            for data in tqdm(self.train_loader, desc='Computing latents'):
                data_dict = to_device(data, self.device)
                samples = data_dict['sample_ori'].float()
                labels = data_dict['class_labels'].long()
                
                mu, _ = self.vae_model.encode(samples)
                
                for i in range(samples.size(0)):
                    class_idx = labels[i].item()
                    class_latents[class_idx].append(mu[i].cpu().numpy())
        
        # Compute statistics
        self.class_statistics = {}
        for cls in range(self.dataset_configs.num_classes):
            if len(class_latents[cls]) > 0:
                latents = np.array(class_latents[cls])
                self.class_statistics[cls] = {
                    'mean': torch.tensor(latents.mean(axis=0)).to(self.device),
                    'std': torch.tensor(latents.std(axis=0)).to(self.device),
                    'count': len(latents)
                }
                print(f"Class {cls} ({self.dataset_configs.class_names[cls]}): {len(latents)} samples")
        
        # Save statistics
        torch.save(self.class_statistics, os.path.join(self.exp_log_dir, 'class_latent_statistics.pt'))
        
        return self.class_statistics
    
    def generate_synthetic_samples(self, target_class, num_samples, output_file=None):
        """Generate synthetic samples for a specific class"""
        print(f"\nGenerating {num_samples} synthetic samples for class {target_class} ({self.dataset_configs.class_names[target_class]})...")
        
        if not hasattr(self, 'class_statistics'):
            self.compute_class_latent_statistics()
        
        if target_class not in self.class_statistics:
            raise ValueError(f"No data available for class {target_class}")
        
        # Generate samples from class-specific latent distribution
        class_mean = self.class_statistics[target_class]['mean']
        class_std = self.class_statistics[target_class]['std']
        
        synthetic_samples = []
        batch_size = self.hparams['batch_size']
        
        self.vae_model.eval()
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                samples = self.vae_model.generate_from_class(
                    current_batch_size, 
                    class_mean, 
                    class_std, 
                    device=self.device
                )
                synthetic_samples.append(samples.cpu())
        
        synthetic_samples = torch.cat(synthetic_samples, dim=0)[:num_samples]
        
        # Create labels
        synthetic_labels = torch.full((num_samples,), target_class, dtype=torch.long)
        
        # Save if output file specified
        if output_file:
            output_path = os.path.join(self.exp_log_dir, output_file)
            torch.save({
                'samples': synthetic_samples.numpy(),
                'labels': synthetic_labels.numpy()
            }, output_path)
            print(f"Saved synthetic samples to {output_path}")
        
        return synthetic_samples, synthetic_labels
    
    def balance_dataset_with_synthetic(self, balance_ratio=1.0):
        """
        Generate synthetic samples to balance the dataset
        balance_ratio: target ratio relative to majority class (1.0 = fully balanced)
        """
        print("\n=== Balancing Dataset with Synthetic Samples ===")
        
        if not hasattr(self, 'class_statistics'):
            self.compute_class_latent_statistics()
        
        # Find majority class count
        max_count = max(self.class_counts.values())
        target_count = int(max_count * balance_ratio)
        
        synthetic_data = {}
        
        for target_class in self.target_classes:
            if target_class not in self.class_counts:
                print(f"Warning: Class {target_class} not found in training data")
                continue
            
            current_count = self.class_counts[target_class]
            num_to_generate = max(0, target_count - current_count)
            
            if num_to_generate > 0:
                print(f"\nClass {target_class} ({self.dataset_configs.class_names[target_class]}):")
                print(f"  Current: {current_count}, Target: {target_count}, Generating: {num_to_generate}")
                
                samples, labels = self.generate_synthetic_samples(
                    target_class, 
                    num_to_generate,
                    output_file=f'synthetic_class_{target_class}_{num_to_generate}samples.pt'
                )
                
                synthetic_data[target_class] = {
                    'samples': samples,
                    'labels': labels
                }
            else:
                print(f"\nClass {target_class} ({self.dataset_configs.class_names[target_class]}): No generation needed (current: {current_count})")
        
        return synthetic_data
    
    def visualize_samples(self, num_real=3, num_synthetic=3, target_class=1):
        """Visualize real vs synthetic samples"""
        print(f"\nGenerating visualization for class {target_class}...")
        
        # Get real samples
        real_samples = []
        for data in self.train_loader:
            data_dict = to_device(data, self.device)
            samples = data_dict['sample_ori'].float()
            labels = data_dict['class_labels'].long()
            
            mask = (labels == target_class)
            if mask.sum() > 0:
                real_samples.append(samples[mask])
            
            if len(real_samples) > 0 and torch.cat(real_samples).size(0) >= num_real:
                break
        
        if len(real_samples) == 0:
            print(f"No real samples found for class {target_class}")
            return
        
        real_samples = torch.cat(real_samples)[:num_real].cpu().numpy()
        
        # Generate synthetic samples
        synthetic_samples, _ = self.generate_synthetic_samples(target_class, num_synthetic)
        synthetic_samples = synthetic_samples.cpu().numpy()
        
        # Plot
        fig, axes = plt.subplots(2, max(num_real, num_synthetic), figsize=(15, 6))
        
        for i in range(num_real):
            axes[0, i].plot(real_samples[i, 0, :])
            axes[0, i].set_title(f'Real Sample {i+1}')
            axes[0, i].set_xlabel('Time')
            axes[0, i].set_ylabel('Amplitude')
            axes[0, i].grid(True, alpha=0.3)
        
        for i in range(num_synthetic):
            axes[1, i].plot(synthetic_samples[i, 0, :])
            axes[1, i].set_title(f'Synthetic Sample {i+1}')
            axes[1, i].set_xlabel('Time')
            axes[1, i].set_ylabel('Amplitude')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.suptitle(f'Real vs Synthetic Samples - Class {target_class} ({self.dataset_configs.class_names[target_class]})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_log_dir, f'real_vs_synthetic_class_{target_class}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {self.exp_log_dir}")


def main():
    parser = argparse.ArgumentParser(description='VAE Trainer for EEG Data Generation')
    
    # Dataset parameters
    parser.add_argument('--dataset', default='sleep_edf', type=str, help='Dataset name')
    parser.add_argument('--data_path', default='data', type=str, help='Path to data directory')
    parser.add_argument('--fold_id', default='0', type=str, help='Fold ID for cross-validation')
    parser.add_argument('--data_percentage', default='100', type=str, help='Percentage of data to use')
    
    # Training parameters
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use')
    parser.add_argument('--mode', default='train', type=str, 
                       help='Mode: train, generate, or both')
    
    # Generation parameters
    parser.add_argument('--target_classes', nargs='+', type=int, default=[1, 3],
                       help='Target classes to generate (default: 1=N1, 3=N3)')
    parser.add_argument('--balance_ratio', type=float, default=0.8,
                       help='Balance ratio (1.0 = fully balanced)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = VAETrainer(args)
    
    # Execute based on mode
    if args.mode == 'train' or args.mode == 'both':
        trainer.train()
    
    if args.mode == 'generate' or args.mode == 'both':
        # Load best model if only generating
        if args.mode == 'generate':
            trainer.load_model()
        
        # Compute latent statistics
        trainer.compute_class_latent_statistics()
        
        # Generate synthetic samples to balance dataset
        synthetic_data = trainer.balance_dataset_with_synthetic(balance_ratio=args.balance_ratio)
        
        # Visualize samples for each target class
        for target_class in args.target_classes:
            try:
                trainer.visualize_samples(num_real=3, num_synthetic=3, target_class=target_class)
            except Exception as e:
                print(f"Could not visualize class {target_class}: {e}")
    
    print("\n=== VAE Training/Generation Complete ===")


if __name__ == "__main__":
    main()

