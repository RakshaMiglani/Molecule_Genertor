"""
Evaluation metrics for molecular generation
Novelty, Uniqueness, Diversity, and Validity
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit import DataStructs
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


class MolecularMetrics:
    """
    Compute molecular generation metrics
    """
    def __init__(self, train_smiles=None):
        """
        train_smiles: list of SMILES from training set (for novelty check)
        """
        self.train_smiles = set(train_smiles) if train_smiles else set()
        
    def is_valid(self, smiles):
        """
        Check if SMILES is valid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def validity(self, generated_smiles):
        """
        Compute validity: fraction of valid SMILES
        """
        valid_count = sum(1 for smi in generated_smiles if self.is_valid(smi))
        return valid_count / len(generated_smiles) if len(generated_smiles) > 0 else 0.0
    
    def uniqueness(self, generated_smiles):
        """
        Compute uniqueness: fraction of unique valid molecules
        """
        valid_smiles = [smi for smi in generated_smiles if self.is_valid(smi)]
        if len(valid_smiles) == 0:
            return 0.0
        
        unique_smiles = set(valid_smiles)
        return len(unique_smiles) / len(valid_smiles)
    
    def novelty(self, generated_smiles):
        """
        Compute novelty: fraction not in training set
        """
        if len(self.train_smiles) == 0:
            return 1.0  # All novel if no training set
        
        valid_smiles = [smi for smi in generated_smiles if self.is_valid(smi)]
        if len(valid_smiles) == 0:
            return 0.0
        
        novel_count = sum(1 for smi in valid_smiles if smi not in self.train_smiles)
        return novel_count / len(valid_smiles)
    
    def diversity(self, generated_smiles, n_samples=1000):
        """
        Compute diversity using average Tanimoto distance
        """
        valid_smiles = [smi for smi in generated_smiles if self.is_valid(smi)]
        
        if len(valid_smiles) < 2:
            return 0.0
        
        # Sample if too many molecules
        if len(valid_smiles) > n_samples:
            valid_smiles = np.random.choice(valid_smiles, n_samples, replace=False)
        
        # Compute Morgan fingerprints
        fingerprints = []
        for smi in valid_smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fingerprints.append(fp)
            except:
                continue
        
        if len(fingerprints) < 2:
            return 0.0
        
        # Compute pairwise Tanimoto similarities
        similarities = []
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                similarities.append(sim)
        
        # Diversity is 1 - average similarity
        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity
        
        return diversity
    
    def compute_properties(self, smiles):
        """
        Compute molecular properties (logP, QED, SAS approximation)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            logp = Descriptors.MolLogP(mol)
            qed = QED.qed(mol)
            
            # SAS approximation
            n_atoms = mol.GetNumAtoms()
            n_rings = Descriptors.RingCount(mol)
            n_hetero = Descriptors.NumHeteroatoms(mol)
            n_rotatable = Descriptors.NumRotatableBonds(mol)
            
            complexity = (
                0.3 * min(n_atoms / 50, 1) +
                0.3 * min(n_rings / 6, 1) +
                0.2 * min(n_hetero / 10, 1) +
                0.2 * min(n_rotatable / 10, 1)
            )
            sas = 1 + complexity * 9
            
            return {'logP': logp, 'qed': qed, 'sas': sas}
        except:
            return None
    
    def property_distribution_match(self, generated_smiles, target_properties):
        """
        Check how well generated molecules match target properties
        
        target_properties: dict with 'logP', 'qed', 'sas' target values
        Returns: dict with mean absolute errors
        """
        valid_smiles = [smi for smi in generated_smiles if self.is_valid(smi)]
        
        if len(valid_smiles) == 0:
            return {'logP_mae': float('inf'), 'qed_mae': float('inf'), 'sas_mae': float('inf')}
        
        errors = {'logP': [], 'qed': [], 'sas': []}
        
        for smi in valid_smiles:
            props = self.compute_properties(smi)
            if props:
                for key in ['logP', 'qed', 'sas']:
                    if key in target_properties:
                        errors[key].append(abs(props[key] - target_properties[key]))
        
        # Compute mean absolute errors
        mae = {}
        for key in ['logP', 'qed', 'sas']:
            if len(errors[key]) > 0:
                mae[f'{key}_mae'] = np.mean(errors[key])
            else:
                mae[f'{key}_mae'] = float('inf')
        
        return mae
    
    def comprehensive_evaluation(self, generated_smiles, target_properties=None):
        """
        Compute all metrics
        """
        results = {
            'validity': self.validity(generated_smiles),
            'uniqueness': self.uniqueness(generated_smiles),
            'novelty': self.novelty(generated_smiles),
            'diversity': self.diversity(generated_smiles),
            'total_generated': len(generated_smiles),
            'valid_molecules': sum(1 for smi in generated_smiles if self.is_valid(smi))
        }
        
        # Property matching if targets provided
        if target_properties:
            prop_match = self.property_distribution_match(generated_smiles, target_properties)
            results.update(prop_match)
        
        return results


def evaluate_model(model, dataset, device, n_samples=1000, 
                   target_properties=None):
    """
    Evaluate model by generating molecules
    
    model: trained VAE model
    dataset: SMILESDataset instance
    n_samples: number of molecules to generate
    target_properties: optional dict with target property values
    """
    model.eval()
    
    print(f"Generating {n_samples} molecules...")
    generated_smiles = []
    
    with torch.no_grad():
        # Sample from prior
        for _ in tqdm(range(n_samples)):
            z = torch.randn(1, model.latent_dim).to(device)
            
            # Decode
            logits = model.decode(z, target_seq=None)
            indices = logits.argmax(dim=-1).squeeze(0)
            
            # Decode to SMILES
            smiles = dataset.decode_smiles(indices)
            generated_smiles.append(smiles)
    
    # Get training SMILES for novelty
    train_smiles = dataset.data['smiles'].tolist()
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = MolecularMetrics(train_smiles=train_smiles)
    results = metrics.comprehensive_evaluation(generated_smiles, target_properties)
    
    return results, generated_smiles


def evaluate_conditional_generation(model, dataset, device, 
                                    target_logp, target_qed, target_sas,
                                    n_samples=100):
    """
    Evaluate conditional generation with specific property targets
    """
    model.eval()
    
    # Normalize target properties
    target_props_norm = dataset.normalize_target_properties(
        target_logp, target_qed, target_sas
    )
    
    print(f"Generating molecules with targets:")
    print(f"  LogP: {target_logp:.2f}")
    print(f"  QED: {target_qed:.2f}")
    print(f"  SAS: {target_sas:.2f}")
    
    generated_smiles = []
    
    with torch.no_grad():
        for _ in tqdm(range(n_samples)):
            # Sample latent vector
            z = torch.randn(1, model.latent_dim).to(device)
            
            # Optionally: optimize z to match properties
            # (This is a simple approach - for better results, use gradient-based optimization)
            
            # Decode
            logits = model.decode(z, target_seq=None)
            indices = logits.argmax(dim=-1).squeeze(0)
            
            # Decode to SMILES
            smiles = dataset.decode_smiles(indices)
            generated_smiles.append(smiles)
    
    # Compute metrics
    train_smiles = dataset.data['smiles'].tolist()
    metrics = MolecularMetrics(train_smiles=train_smiles)
    
    target_props_dict = {'logP': target_logp, 'qed': target_qed, 'sas': target_sas}
    results = metrics.comprehensive_evaluation(generated_smiles, target_props_dict)
    
    return results, generated_smiles


def print_evaluation_results(results):
    """
    Pretty print evaluation results
    """
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"\nGeneration Statistics:")
    print(f"  Total Generated: {results['total_generated']}")
    print(f"  Valid Molecules: {results['valid_molecules']}")
    
    print(f"\nQuality Metrics:")
    print(f"  Validity:    {results['validity']:.4f} ({results['validity']*100:.2f}%)")
    print(f"  Uniqueness:  {results['uniqueness']:.4f} ({results['uniqueness']*100:.2f}%)")
    print(f"  Novelty:     {results['novelty']:.4f} ({results['novelty']*100:.2f}%)")
    print(f"  Diversity:   {results['diversity']:.4f}")
    
    if 'logP_mae' in results:
        print(f"\nProperty Matching (MAE):")
        print(f"  LogP MAE: {results['logP_mae']:.4f}")
        print(f"  QED MAE:  {results['qed_mae']:.4f}")
        print(f"  SAS MAE:  {results['sas_mae']:.4f}")
    
    print("="*50 + "\n")


if __name__ == '__main__':
    # Test metrics
    test_smiles = [
        'CCO',
        'CCCC',
        'c1ccccc1',
        'CC(C)O',
        'invalid_smiles',
        'CCO',  # Duplicate
    ]
    
    train_smiles = ['CCO', 'CCCC']
    
    metrics = MolecularMetrics(train_smiles=train_smiles)
    
    print("Test SMILES:")
    for smi in test_smiles:
        print(f"  {smi}")
    
    results = metrics.comprehensive_evaluation(test_smiles)
    print_evaluation_results(results)