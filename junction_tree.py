"""
Junction Tree VAE for combining molecular fragments
Used to merge molecules satisfying different property constraints
"""

import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS
from collections import defaultdict


class MolecularGraph:
    """
    Molecular graph representation for junction tree
    """
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
    def get_fragments(self):
        """
        Decompose molecule into BRICS fragments
        """
        if self.mol is None:
            return []
        
        # Use BRICS decomposition
        fragments = BRICS.BRICSDecompose(self.mol)
        return list(fragments)
    
    def get_atoms_features(self):
        """
        Get atom features for each atom
        """
        features = []
        for atom in self.mol.GetAtoms():
            features.append([
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetIsAromatic())
            ])
        return features


class JunctionTreeNode:
    """
    Node in junction tree representing a molecular fragment
    """
    def __init__(self, smiles, idx):
        self.smiles = smiles
        self.idx = idx
        self.neighbors = []
        self.mol = Chem.MolFromSmiles(smiles)
        
    def add_neighbor(self, node):
        self.neighbors.append(node)


class JunctionTree:
    """
    Junction tree representation of a molecule
    """
    def __init__(self, smiles):
        self.smiles = smiles
        self.nodes = []
        self.build_tree()
        
    def build_tree(self):
        """
        Build junction tree from SMILES
        """
        mol_graph = MolecularGraph(self.smiles)
        fragments = mol_graph.get_fragments()
        
        # Create nodes for each fragment
        for idx, frag_smiles in enumerate(fragments):
            node = JunctionTreeNode(frag_smiles, idx)
            self.nodes.append(node)
        
        # Connect fragments (simplified - in production, use proper bond analysis)
        for i in range(len(self.nodes) - 1):
            self.nodes[i].add_neighbor(self.nodes[i + 1])
            self.nodes[i + 1].add_neighbor(self.nodes[i])


class JunctionTreeVAE(nn.Module):
    """
    Junction Tree VAE for molecular generation
    Combines fragments from different molecules
    """
    def __init__(self, vocab_size, hidden_dim=128, latent_dim=64):
        super(JunctionTreeVAE, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Tree encoder (processes junction tree structure)
        self.tree_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Graph encoder (processes molecular graph)
        self.graph_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoder
        self.decoder = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        
    def encode_tree(self, tree):
        """
        Encode junction tree structure
        """
        # Simplified encoding - in production, use message passing
        node_features = []
        for node in tree.nodes:
            # Create feature vector for node (fragment)
            feat = torch.randn(self.hidden_dim)  # Placeholder
            node_features.append(feat)
        
        if len(node_features) == 0:
            return torch.zeros(1, self.hidden_dim)
        
        node_features = torch.stack(node_features).unsqueeze(0)
        _, hidden = self.tree_encoder(node_features)
        return hidden.squeeze(0)
    
    def forward(self, tree):
        """
        Forward pass through junction tree VAE
        """
        # Encode tree
        tree_encoding = self.encode_tree(tree)
        
        # Simple pass-through for now
        return tree_encoding


class MolecularMerger:
    """
    Merge molecules based on property constraints using junction trees
    """
    def __init__(self):
        self.fragment_library = defaultdict(list)
        
    def decompose_molecule(self, smiles):
        """
        Decompose molecule into fragments
        """
        try:
            mol_graph = MolecularGraph(smiles)
            fragments = mol_graph.get_fragments()
            return fragments
        except:
            return []
    
    def build_fragment_library(self, molecules_by_property):
        """
        Build library of fragments from molecules optimized for each property
        molecules_by_property: dict of {property_name: [smiles]}
        """
        for prop_name, smiles_list in molecules_by_property.items():
            for smiles in smiles_list:
                fragments = self.decompose_molecule(smiles)
                for frag in fragments:
                    self.fragment_library[prop_name].append(frag)
        
        # Remove duplicates
        for prop_name in self.fragment_library:
            self.fragment_library[prop_name] = list(set(self.fragment_library[prop_name]))
    
    def combine_fragments(self, fragments):
        """
        Combine fragments into a valid molecule
        Uses BRICS recombination
        """
        if len(fragments) == 0:
            return None
        
        if len(fragments) == 1:
            return fragments[0]
        
        # Try to combine fragments using BRICS
        try:
            # Convert fragments to molecules
            frag_mols = [Chem.MolFromSmiles(f) for f in fragments if Chem.MolFromSmiles(f)]
            
            if len(frag_mols) == 0:
                return None
            
            # Simple approach: try pairwise combinations
            combined = frag_mols[0]
            for frag_mol in frag_mols[1:]:
                # Try to combine (simplified - production would use proper BRICS synthesis)
                try:
                    combined = Chem.CombineMols(combined, frag_mol)
                except:
                    continue
            
            if combined:
                smiles = Chem.MolToSmiles(combined)
                # Validate
                if Chem.MolFromSmiles(smiles):
                    return smiles
            
            return None
        except:
            return None
    
    def merge_molecules_for_constraints(self, molecules_dict, target_properties):
        """
        Merge molecules to satisfy multiple property constraints
        
        molecules_dict: {'logP': [mol1, mol2], 'qed': [mol3], 'sas': [mol4]}
        target_properties: {'logP': 3.5, 'qed': 0.7, 'sas': 2.5}
        
        Returns: candidate merged molecules
        """
        # Build fragment library
        self.build_fragment_library(molecules_dict)
        
        # Select fragments from each property
        selected_fragments = []
        for prop_name, target_val in target_properties.items():
            if prop_name in self.fragment_library:
                # Select some fragments (simplified selection)
                frags = self.fragment_library[prop_name]
                if len(frags) > 0:
                    # Take first few fragments as candidates
                    selected_fragments.extend(frags[:min(3, len(frags))])
        
        # Try to combine fragments
        candidates = []
        
        # Try different combinations
        for i in range(min(5, len(selected_fragments))):
            for j in range(i + 1, min(5, len(selected_fragments))):
                merged = self.combine_fragments([selected_fragments[i], selected_fragments[j]])
                if merged and Chem.MolFromSmiles(merged):
                    candidates.append(merged)
        
        return list(set(candidates))  # Remove duplicates


def weighted_latent_interpolation(z_list, weights):
    """
    Interpolate latent vectors based on property weights
    
    z_list: list of latent vectors for molecules optimized for each property
    weights: list of weights for each property
    """
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()  # Normalize
    
    # Weighted average in latent space
    z_combined = sum(w * z for w, z in zip(weights, z_list))
    return z_combined


def test_merger():
    """
    Test molecular merger
    """
    # Example molecules optimized for different properties
    molecules = {
        'logP': ['CCO', 'CCCC'],  # High logP
        'qed': ['CC(C)Cc1ccc(cc1)C(C)C(=O)O'],  # Drug-like
        'sas': ['CC', 'CCC']  # Easy to synthesize
    }
    
    merger = MolecularMerger()
    
    # Test fragment decomposition
    for prop, smiles_list in molecules.items():
        print(f"\n{prop} molecules:")
        for smi in smiles_list:
            frags = merger.decompose_molecule(smi)
            print(f"  {smi} -> {frags}")
    
    # Test merging
    target_props = {'logP': 3.0, 'qed': 0.7, 'sas': 2.0}
    candidates = merger.merge_molecules_for_constraints(molecules, target_props)
    print(f"\nMerged candidates: {candidates}")


if __name__ == '__main__':
    test_merger()