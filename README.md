# T3N5OR: Robust Sleep Stage Detection from Scarce & Imbalanced EEG Data

**Team Name:** T3N5OR

**Project Title:** Robust Sleep Stage Detection from Scarce & Imbalanced EEG Data using Generative Modelling and Semi-Supervised Learning

---

## 📌 Project Overview

This project focuses on detecting sleep stages using the Sleep-EDF (Sleep Cassette) dataset, which contains EEG recordings from 20 healthy adults. The dataset is inherently imbalanced, with N1 and N3 stages being underrepresented. Our solution addresses this by leveraging Generative Modelling (VAE) to synthesize minority-class data and Semi-Supervised Learning (SSL) to utilize unlabeled data effectively.

### Key Features

- **Robust Baseline:** 1D-CNN classifier with class-weighted and focal loss.
- **Generative Modelling:** Uses a Variational Autoencoder (VAE) to generate realistic synthetic EEG data for minority classes (N1/N3).
- **Semi-Supervised Learning (SSL):** Implements algorithms like SimCLR to improve generalization and label efficiency on scarce data.

---

## 📊 Dataset

- **Source:** Sleep-EDF (Original Subset)
- **Link:** [PhysioNet Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)
- **Details:** Recordings from 20 healthy adults (25-34 years) over 39 nights, annotated with W, N1, N2, N3, REM stages.

---

## ⚙️ Installation

Ensure you have Python 3 installed. Run the following commands to install the required dependencies:

```bash
pip install torch torchvision
pip install numpy
pip install pandas
pip install scikit-learn
pip install mne
pip install einops
```

---

## 🚀 Project Structure & Usage

### 1. Preprocessing

First, download the Sleep-EDF dataset (Sleep Cassette) and run the preprocessing script to convert the data into a usable format.

```bash
# Install MNE if not already installed
pip3 install mne

# Create directory for output files
mkdir my_npz_files

# Run preprocessing (Replace 'path/to/sleep-cassette' with your actual data path)
python3 PREPROCESSING\ FILES\ /data_preprocessing/sleep-edf/preprocess_sleep_edf.py --data_dir /Users/taruns/Downloads/TS-TCC-main/sleep-cassette --output_dir my_npz_files
```

### 2. Main Setup (Data Splitting)

Organize the preprocessed files and generate k-fold splits.

```bash
# Create numpy directory and move files
mkdir MAIN\ FILE/np
mv /Users/taruns/Downloads/TS-TCC-main/my_npz_files/*.npz MAIN\ FILE/np/

# Generate K-Fold splits
cd MAIN\ FILE/
python3 split_k-fold_and_few_labels.py

# Organize into data directory
mkdir -p data/sleep_edf
mv folds_data/*.pt data/sleep_edf/
```

---

## 🧪 Running Experiments

### 1. Baseline Model (No Augmentation)

Train the supervised 1D-CNN baseline model.

```bash
cd MAIN\ FILE/
python3 main.py --experiment_description "Baseline" --train_mode supervised_100per --fold_id 0 --device cpu
```

### 2. VAE Augmented Baseline

Train a VAE to generate synthetic data for minority classes, augment the dataset, and retrain the model.

#### Step A: Train the VAE

```bash
python3 vae_trainer.py --dataset sleep_edf --fold_id 0 --device cpu --mode both --target_classes 1 3 --balance_ratio 0.8
```

#### Step B: Generate synthetic data

```bash
python3 augment_data_with_vae.py --mode single --original_path data/sleep_edf/train_0_100per.pt --vae_dir experiments_logs_sleep_edf/VAE_Generation/Fold_0/ --output_path data/sleep_edf_augmented/train_0_100per_augmented.pt
```

#### Step C: Train model on augmented data

```bash
python3 main.py --experiment_description "Baseline_VAE" --train_mode supervised_100per --fold_id 0 --device cpu --data_path data/sleep_edf_augmented
```

### 3. Semi-Supervised Learning (SimCLR)

Leverage unlabeled data using SimCLR.

#### Pretrain

```bash
python3 main.py --experiment_description "SSL_SimCLR" --run_description "Pretrain" --ssl_method simclr --train_mode ssl --fold_id 0 --device cpu
```

#### Finetune (e.g., on 5% labeled data)

```bash
python3 main.py --experiment_description "SSL_SimCLR" --run_description "Finetune" --ssl_method simclr --train_mode ft_5per --fold_id 0 --device cpu
```

### 4. SSL + VAE Augmentation

Combine generative augmentation with SSL pretraining.

#### Pretrain on Augmented Data

```bash
python3 main.py --experiment_description "SSL_SimCLR_VAE" --run_description "Pretrain" --ssl_method simclr --train_mode ssl --fold_id 0 --device cpu --data_path data/sleep_edf_augmented
```

#### Finetune on Augmented Data

```bash
python3 main.py --experiment_description "SSL_SimCLR_VAE" --run_description "Finetune" --ssl_method simclr --train_mode ft_5per --fold_id 0 --device cpu --data_path data/sleep_edf_augmented
```

---

## 👥 Team Members

| Name | SRN |
|------|-----|
| Adarsh R Menon | PES1UG23AM016 |
| Akshay P Shetti | PES1UG23AM039 |
| Tarun S | PES1UG23AM919 |
| Adityaa Kumar H | PES1UG23AM025 |

---


---

## 📁 Project Structure

```
AFML COURSE PROJECT/
├── MAIN FILE/              # Main training and evaluation scripts
│   ├── main.py            # Main training script
│   ├── vae_trainer.py     # VAE training script
│   ├── augment_data_with_vae.py  # Data augmentation script
│   ├── split_k-fold_and_few_labels.py  # K-fold splitting
│   ├── models/            # Model architectures
│   ├── dataloader/        # Data loading utilities
│   ├── configs/           # Configuration files
│   └── folds_data/        # K-fold split data
├── PREPROCESSING FILES/   # Data preprocessing scripts
│   ├── data_preprocessing/  # Dataset-specific preprocessing
│   └── config_files/      # Preprocessing configurations
└── OUTPUT/                # Results and analysis outputs
```

---


---

## 📝 License

This project is part of an academic course project.

