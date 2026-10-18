"""
Data Download and Preprocessing Script for ZINC Dataset
Downloads ZINC dataset and computes molecular properties
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import requests
from tqdm import tqdm
import os
import pickle


def download_zinc_dataset(output_path='data/zinc_dataset.csv', num_molecules=250000):
    """
    Download ZINC dataset from GitHub or generate from SMILES
    """
    print(f"Downloading ZINC dataset with {num_molecules} molecules...")
    
    # Try multiple sources
    urls = [
        'https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv',
        'https://raw.githubusercontent.com/molecularsets/moses/master/data/dataset_v1.csv'
    ]
    
    for url in urls:
        try:
            print(f"Trying to download from: {url}")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded to {output_path}")
                return True
        except Exception as e:
            print(f"Failed to download from {url}: {str(e)}")
            continue
    
    print("Could not download from default sources. Please manually download ZINC dataset.")
    print("Expected format: CSV with 'smiles' column")
    return False


def calculate_properties(smiles):
    """
    Calculate LogP, QED, and SAS for a SMILES string
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None
        
        # LogP (Lipophilicity)
        logp = Descriptors.MolLogP(mol)
        
        # QED (Quantitative Estimate of Drug-likeness)
        qed = QED.qed(mol)
        
        # SAS (Synthetic Accessibility Score) - approximation
        # RDKit doesn't have built-in SAS, so we use a simple approximation
        # For full implementation, you'd need the SAS score calculator
        sas = calculate_sas_approximation(mol)
        
        return logp, qed, sas
    except:
        return None, None, None


def calculate_sas_approximation(mol):
    """
    Approximate SAS score based on molecular complexity
    Range: 1 (easy to synthesize) to 10 (difficult)
    This is a simplified version - for production, use the full SAS implementation
    """
    # Simple complexity metrics
    n_atoms = mol.GetNumAtoms()
    n_rings = Descriptors.RingCount(mol)
    n_hetero = Descriptors.NumHeteroatoms(mol)
    n_rotatable = Descriptors.NumRotatableBonds(mol)
    
    # Normalized complexity score
    complexity = (
        0.3 * min(n_atoms / 50, 1) +
        0.3 * min(n_rings / 6, 1) +
        0.2 * min(n_hetero / 10, 1) +
        0.2 * min(n_rotatable / 10, 1)
    )
    
    # Scale to 1-10 range (inverted so lower is easier)
    sas = 1 + complexity * 9
    return sas


def preprocess_zinc_dataset(input_path='data/zinc_dataset.csv', 
                            output_path='data/processed/zinc_processed.csv',
                            max_length=120):
    """
    Preprocess ZINC dataset: validate SMILES and compute properties
    """
    print("Loading ZINC dataset...")
    df = pd.read_csv(input_path)
    
    # Ensure we have a SMILES column (try different common names)
    smiles_col = None
    for col in ['smiles', 'SMILES', 'smile', 'SMILE']:
        if col in df.columns:
            smiles_col = col
            break
    
    if smiles_col is None:
        raise ValueError("Could not find SMILES column in dataset")
    
    df = df.rename(columns={smiles_col: 'smiles'})
    
    print(f"Processing {len(df)} molecules...")
    
    valid_data = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row['smiles']
        
        # Skip if too long
        if len(smiles) > max_length:
            continue
        
        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        # Canonicalize SMILES
        canonical_smiles = Chem.MolToSmiles(mol)
        
        # Calculate or use existing properties
        if 'logP' in df.columns or 'logp' in df.columns:
            logp = row.get('logP', row.get('logp', None))
            qed = row.get('qed', row.get('QED', None))
            sas = row.get('SA', row.get('sas', row.get('SAS', None)))
        else:
            logp, qed, sas = calculate_properties(canonical_smiles)
        
        if logp is not None and qed is not None and sas is not None:
            valid_data.append({
                'smiles': canonical_smiles,
                'logP': float(logp),
                'qed': float(qed),
                'sas': float(sas)
            })
    
    # Create processed dataframe
    processed_df = pd.DataFrame(valid_data)
    
    print(f"\nProcessed Statistics:")
    print(f"Valid molecules: {len(processed_df)}")
    print(f"LogP range: [{processed_df['logP'].min():.2f}, {processed_df['logP'].max():.2f}]")
    print(f"QED range: [{processed_df['qed'].min():.2f}, {processed_df['qed'].max():.2f}]")
    print(f"SAS range: [{processed_df['sas'].min():.2f}, {processed_df['sas'].max():.2f}]")
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    
    return processed_df


def create_vocabulary(df, output_path='data/processed/vocab.pkl'):
    """
    Create character vocabulary from SMILES strings
    """
    print("Creating vocabulary...")
    
    # Collect all unique characters
    chars = set()
    for smiles in df['smiles']:
        chars.update(smiles)
    
    # Sort for consistency
    chars = sorted(list(chars))
    
    # Add special tokens
    special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
    vocab = special_tokens + chars
    
    # Create mappings
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    idx_to_char = {i: c for i, c in enumerate(vocab)}
    
    vocab_dict = {
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': len(vocab),
        'max_length': df['smiles'].str.len().max()
    }
    
    # Save vocabulary
    with open(output_path, 'wb') as f:
        pickle.dump(vocab_dict, f)
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Max SMILES length: {vocab_dict['max_length']}")
    print(f"Saved vocabulary to {output_path}")
    
    return vocab_dict


def main():
    """
    Main preprocessing pipeline
    """
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Download dataset
    zinc_path = 'data/zinc_dataset.csv'
    if not os.path.exists(zinc_path):
        success = download_zinc_dataset(zinc_path)
        if not success:
            print("\nPlease manually download ZINC dataset and place it at 'data/zinc_dataset.csv'")
            print("Required columns: 'smiles' (and optionally 'logP', 'qed', 'sas')")
            return
    
    # Preprocess dataset
    processed_path = 'data/processed/zinc_processed.csv'
    if not os.path.exists(processed_path):
        df = preprocess_zinc_dataset(zinc_path, processed_path)
    else:
        print(f"Loading existing processed data from {processed_path}")
        df = pd.read_csv(processed_path)
    
    # Create vocabulary
    vocab_path = 'data/processed/vocab.pkl'
    if not os.path.exists(vocab_path):
        create_vocabulary(df, vocab_path)
    else:
        print(f"Vocabulary already exists at {vocab_path}")
    
    print("\n✓ Data preprocessing complete!")
    print(f"  - Processed dataset: {processed_path}")
    print(f"  - Vocabulary: {vocab_path}")


if __name__ == '__main__':
    main()