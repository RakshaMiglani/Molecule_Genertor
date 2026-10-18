"""
Dataset and data utilities for molecular VAE
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
import os


class SMILESDataset(Dataset):
    """
    Dataset for SMILES strings with properties
    """
    def __init__(self, csv_path, vocab_path, max_length=120):
        self.data = pd.read_csv(csv_path)
        
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        self.char_to_idx = self.vocab['char_to_idx']
        self.idx_to_char = self.vocab['idx_to_char']
        self.max_length = max_length
        
        # Normalize properties
        self.property_means = {
            'logP': self.data['logP'].mean(),
            'qed': self.data['qed'].mean(),
            'sas': self.data['sas'].mean()
        }
        self.property_stds = {
            'logP': self.data['logP'].std(),
            'qed': self.data['qed'].std(),
            'sas': self.data['sas'].std()
        }
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row['smiles']
        
        # Encode SMILES
        encoded = self.encode_smiles(smiles)
        
        # Get properties (normalized)
        properties = torch.tensor([
            (row['logP'] - self.property_means['logP']) / self.property_stds['logP'],
            (row['qed'] - self.property_means['qed']) / self.property_stds['qed'],
            (row['sas'] - self.property_means['sas']) / self.property_stds['sas']
        ], dtype=torch.float32)
        
        return {
            'encoded': encoded,
            'properties': properties,
            'smiles': smiles
        }
    
    def encode_smiles(self, smiles):
        """
        Encode SMILES string to indices
        """
        # Add START token
        encoded = [self.char_to_idx['<START>']]
        
        # Encode characters
        for char in smiles:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                encoded.append(self.char_to_idx['<UNK>'])
        
        # Add END token
        encoded.append(self.char_to_idx['<END>'])
        
        # Pad to max_length
        if len(encoded) < self.max_length:
            encoded += [self.char_to_idx['<PAD>']] * (self.max_length - len(encoded))
        else:
            encoded = encoded[:self.max_length]
        
        return torch.tensor(encoded, dtype=torch.long)
    
    def decode_smiles(self, indices):
        """
        Decode indices back to SMILES string
        """
        smiles = []
        for idx in indices:
            idx = idx.item() if torch.is_tensor(idx) else idx
            char = self.idx_to_char.get(idx, '')
            
            if char in ['<PAD>', '<START>', '<END>', '<UNK>']:
                if char == '<END>':
                    break
                continue
            
            smiles.append(char)
        
        return ''.join(smiles)
    
    def denormalize_properties(self, properties):
        """
        Convert normalized properties back to original scale
        """
        if torch.is_tensor(properties):
            properties = properties.cpu().numpy()
        
        denorm = np.zeros_like(properties)
        denorm[0] = properties[0] * self.property_stds['logP'] + self.property_means['logP']
        denorm[1] = properties[1] * self.property_stds['qed'] + self.property_means['qed']
        denorm[2] = properties[2] * self.property_stds['sas'] + self.property_means['sas']
        
        return denorm
    
    def normalize_target_properties(self, logp, qed, sas):
        """
        Normalize target properties for generation
        """
        norm_logp = (logp - self.property_means['logP']) / self.property_stds['logP']
        norm_qed = (qed - self.property_means['qed']) / self.property_stds['qed']
        norm_sas = (sas - self.property_means['sas']) / self.property_stds['sas']
        
        return torch.tensor([norm_logp, norm_qed, norm_sas], dtype=torch.float32)


def create_dataloaders(csv_path, vocab_path, batch_size=64, 
                      train_split=0.8, val_split=0.1):
    """
    Create train, validation, and test dataloaders
    """
    dataset = SMILESDataset(csv_path, vocab_path)
    
    # Split dataset
    n_total = len(dataset)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=2, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, dataset


def collate_fn(batch):
    """
    Custom collate function for dataloader
    """
    encoded = torch.stack([item['encoded'] for item in batch])
    properties = torch.stack([item['properties'] for item in batch])
    smiles = [item['smiles'] for item in batch]
    
    return {
        'encoded': encoded,
        'properties': properties,
        'smiles': smiles
    }


if __name__ == '__main__':
    # Test dataset
    csv_path = '../data/processed/zinc_processed.csv'
    vocab_path = '../data/processed/vocab.pkl'
    
    if not (os.path.exists(csv_path) and os.path.exists(vocab_path)):
        print("Please run data/download_and_preprocess.py first")
    else:
        dataset = SMILESDataset(csv_path, vocab_path)
        print(f"Dataset size: {len(dataset)}")
        print(f"Vocab size: {dataset.vocab['vocab_size']}")
        
        # Test sample
        sample = dataset[0]
        print(f"\nSample:")
        print(f"SMILES: {sample['smiles']}")
        print(f"Encoded shape: {sample['encoded'].shape}")
        print(f"Properties: {sample['properties']}")
        
        # Test decoding
        decoded = dataset.decode_smiles(sample['encoded'])
        print(f"Decoded: {decoded}")