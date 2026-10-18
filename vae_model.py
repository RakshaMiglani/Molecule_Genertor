"""
Variational Autoencoder for Molecular Generation
Includes property prediction heads and beta-annealing for KL loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


class MolecularVAE(nn.Module):
    """
    VAE for molecular generation with property prediction
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 latent_dim=128, max_length=120, num_properties=3):
        super(MolecularVAE, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.num_properties = num_properties
        
        # Encoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, 
                                   num_layers=2, bidirectional=True, dropout=0.2)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Property prediction heads
        self.property_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_properties)
        )
        
        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, 
                                   num_layers=2, dropout=0.2)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def encode(self, x):
        """
        Encode input sequences to latent space
        x: [batch_size, seq_len]
        """
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        _, hidden = self.encoder_gru(embedded)  # hidden: [4, batch_size, hidden_dim]
        
        # Concatenate forward and backward final hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch_size, hidden_dim*2]
        
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, target_seq=None):
        """
        Decode latent vectors to sequences
        z: [batch_size, latent_dim]
        target_seq: [batch_size, seq_len] (for teacher forcing during training)
        """
        batch_size = z.size(0)
        
        # Initialize hidden state from latent vector
        hidden = self.latent_to_hidden(z).unsqueeze(0)  # [1, batch_size, hidden_dim]
        hidden = hidden.repeat(2, 1, 1)  # [2, batch_size, hidden_dim] for 2 layers
        
        if target_seq is not None:
            # Teacher forcing mode (training)
            embedded = self.embedding(target_seq)  # [batch_size, seq_len, embedding_dim]
            output, _ = self.decoder_gru(embedded, hidden)
            logits = self.fc_out(output)  # [batch_size, seq_len, vocab_size]
        else:
            # Autoregressive mode (inference)
            logits = []
            input_token = torch.ones(batch_size, 1, dtype=torch.long, 
                                    device=z.device)  # <START> token (idx=1)
            
            for _ in range(self.max_length):
                embedded = self.embedding(input_token)
                output, hidden = self.decoder_gru(embedded, hidden)
                step_logits = self.fc_out(output.squeeze(1))
                logits.append(step_logits)
                
                # Get next input token (greedy decoding)
                input_token = step_logits.argmax(dim=-1, keepdim=True)
                
            logits = torch.stack(logits, dim=1)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def predict_properties(self, z):
        """
        Predict molecular properties from latent space
        Returns: [batch_size, num_properties] - [logP, qed, sas]
        """
        return self.property_predictor(z)
    
    def forward(self, x, beta=1.0):
        """
        Forward pass with KL annealing
        x: [batch_size, seq_len]
        beta: KL weight for annealing
        """
        # Encode
        mu, logvar = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        logits = self.decode(z, target_seq=x)
        
        # Predict properties
        predicted_properties = self.predict_properties(z)
        
        return logits, mu, logvar, predicted_properties, z


class VAELoss(nn.Module):
    """
    Combined loss for VAE with property prediction
    """
    def __init__(self, vocab_size, property_weights=None):
        super(VAELoss, self).__init__()
        self.vocab_size = vocab_size
        
        # Weights for multi-task learning
        if property_weights is None:
            property_weights = [1.0, 1.0, 1.0]  # Equal weights for logP, qed, sas
        self.property_weights = property_weights
        
    def forward(self, logits, targets, mu, logvar, pred_properties, 
                true_properties, beta=1.0):
        """
        Compute combined loss
        """
        batch_size = logits.size(0)
        
        # Reconstruction loss (Cross Entropy)
        recon_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=0,  # Ignore padding
            reduction='sum'
        ) / batch_size
        
        # KL divergence with annealing
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        kl_loss = beta * kl_loss
        
        # Property prediction loss (MSE for regression)
        property_loss = 0
        for i in range(pred_properties.size(1)):
            prop_loss = F.mse_loss(pred_properties[:, i], true_properties[:, i])
            property_loss += self.property_weights[i] * prop_loss
        
        # Total loss
        total_loss = recon_loss + kl_loss + property_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'property_loss': property_loss
        }


class BetaScheduler:
    """
    Beta annealing scheduler to prevent posterior collapse
    """
    def __init__(self, mode='cyclical', n_epochs=100, start=0.0, stop=1.0, 
                 n_cycles=4, ratio=0.5):
        self.mode = mode
        self.n_epochs = n_epochs
        self.start = start
        self.stop = stop
        self.n_cycles = n_cycles
        self.ratio = ratio
        
    def get_beta(self, epoch):
        """
        Get beta value for current epoch
        """
        if self.mode == 'monotonic':
            # Linear increase
            beta = min(self.start + (self.stop - self.start) * epoch / self.n_epochs, 
                      self.stop)
        
        elif self.mode == 'cyclical':
            # Cyclical annealing (ICLR 2019)
            cycle_length = self.n_epochs / self.n_cycles
            cycle_position = epoch % cycle_length
            
            if cycle_position < cycle_length * self.ratio:
                # Increase phase
                beta = self.start + (self.stop - self.start) * \
                       (cycle_position / (cycle_length * self.ratio))
            else:
                # Constant phase
                beta = self.stop
        
        elif self.mode == 'constant':
            beta = self.stop
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return beta


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model
    vocab_size = 50
    batch_size = 32
    seq_len = 100
    
    model = MolecularVAE(vocab_size=vocab_size)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    properties = torch.randn(batch_size, 3)
    
    logits, mu, logvar, pred_props, z = model(x, beta=0.5)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Predicted properties shape: {pred_props.shape}")
    
    # Test loss
    loss_fn = VAELoss(vocab_size)
    losses = loss_fn(logits, x, mu, logvar, pred_props, properties, beta=0.5)
    print(f"\nLosses: {losses}")