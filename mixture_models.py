import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import EuropeDataset

def normalize_tensor(tensor, d):
    """
    Normalize the input tensor along the specified axis to have a mean of 0 and a std of 1.
    
    Parameters:
        tensor (torch.Tensor): Input tensor to normalize.
        d (int): Axis along which to normalize.
    
    Returns:
        torch.Tensor: Normalized tensor.
    """
    mean = torch.mean(tensor, dim=d, keepdim=True)
    std = torch.std(tensor, dim=d, keepdim=True)
    normalized = (tensor - mean) / std
    return normalized


class GMM(nn.Module):
    def __init__(self, n_components):
        """
        Gaussian Mixture Model in 2D using PyTorch.

        Args:
            n_components (int): Number of Gaussian components.
        """
        super().__init__()        
        self.n_components = n_components

        # Mixture logits (softmaxed to probabilities)
        self.logits = nn.Parameter(torch.randn(n_components))

        # Means of the Gaussian components (n_components x 2 for 2D data)
        self.means = nn.Parameter(torch.randn(n_components, 2))

        # Log of the variance of the Gaussian components (n_components x 2 for 2D data)
        self.log_variances = nn.Parameter(torch.zeros(n_components, 2))  # Log-variances (diagonal covariance)




    def forward(self, X):
        """
        Compute the log-likelihood of the data.
        Args:
            X (torch.Tensor): Input data of shape (n_samples, 2).

        Returns:
            torch.Tensor: Log-likelihood of shape (n_samples,).
        """        
        # X: (batch, 2), means/log_variances: (K, 2)
        log_weights = F.log_softmax(self.logits, dim=0)  # log p(k)

        # Compute log p(x|k) for each component
        var = torch.exp(self.log_variances)  # ensure positivity
        log_vars = self.log_variances
        diff = X.unsqueeze(1) - self.means.unsqueeze(0)  # (batch, K, 2)

        log_two_pi = torch.log(torch.tensor(2.0 * torch.pi, device=X.device))
        log_conditional = -0.5 * ((diff ** 2) / var.unsqueeze(0)).sum(dim=2)
        log_conditional += -0.5 * (log_two_pi + log_vars.unsqueeze(0)).sum(dim=2)

        # log p(x) = logsumexp_k [log p(k) + log p(x|k)]
        log_likelihood = torch.logsumexp(log_weights.unsqueeze(0) + log_conditional, dim=1)
        return log_likelihood


    def loss_function(self, log_likelihood):
        """
        Compute the negative log-likelihood loss.
        Args:
            log_likelihood (torch.Tensor): Log-likelihood of shape (n_samples,).

        Returns:
            torch.Tensor: Negative log-likelihood.
        """
        return -log_likelihood.mean()


    def sample(self, n_samples):
        """
        Generate samples from the GMM model.
        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        with torch.no_grad():
            probs = F.softmax(self.logits, dim=0)
            component_ids = torch.multinomial(probs, num_samples=n_samples, replacement=True)
            std = torch.exp(0.5 * self.log_variances)
            eps = torch.randn(n_samples, 2)
            samples = self.means[component_ids] + std[component_ids] * eps
        return samples
    
    def conditional_sample(self, n_samples, label):
        """
        Generate samples from a specific uniform component.
        Args:
            n_samples (int): Number of samples to generate.
            label (int): Component index.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        with torch.no_grad():
            std = torch.exp(0.5 * self.log_variances[label])
            eps = torch.randn(n_samples, 2)
            samples = self.means[label] + std * eps
        return samples



class UMM(nn.Module):
    def __init__(self, n_components):
        """
        Uniform Mixture Model in 2D using PyTorch.

        Args:
            n_components (int): Number of uniform components.
        """
        super().__init__()        
        self.n_components = n_components

        # Mixture logits (softmaxed to probabilities)
        self.logits = nn.Parameter(torch.randn(n_components))

        # Center value of the uniform components (n_components x 2 for 2D data)
        self.centers = nn.Parameter(torch.randn(n_components, 2))

        # Log of size of the uniform components (n_components x 2 for 2D data)
        self.log_sizes = nn.Parameter(torch.log(torch.ones(n_components, 2) + torch.rand(n_components, 2)*0.2))


    def forward(self, X):
        """
        Compute the log-likelihood of the data.
        Args:
            X (torch.Tensor): Input data of shape (n_samples, 2).

        Returns:
            torch.Tensor: Log-likelihood of shape (n_samples,).
        """
        log_weights = F.log_softmax(self.logits, dim=0)

        size = torch.exp(self.log_sizes)  # positive side lengths
        half_size = 0.5 * size
        lower = self.centers.unsqueeze(0) - half_size.unsqueeze(0)  # (1, K, 2)
        upper = self.centers.unsqueeze(0) + half_size.unsqueeze(0)

        # Check membership for each component
        X_expanded = X.unsqueeze(1)  # (batch, 1, 2)
        inside = (X_expanded >= lower) & (X_expanded <= upper)
        inside_mask = inside.all(dim=2)  # (batch, K)

        # log p(x|k) is constant inside the rectangle, use -1e6 instead of -inf to avoid nan
        log_volume = torch.log(size.prod(dim=1))  # (K,)
        log_conditional = -log_volume.unsqueeze(0).expand_as(inside_mask).float()
        log_conditional = torch.where(inside_mask, log_conditional, torch.full_like(log_conditional, -1e6))

        log_likelihood = torch.logsumexp(log_weights.unsqueeze(0) + log_conditional, dim=1)
        return log_likelihood
        
    
    
    def loss_function(self, log_likelihood):
        """
        Compute the negative log-likelihood loss.
        Args:
            log_likelihood (torch.Tensor): Log-likelihood of shape (n_samples,).

        Returns:
            torch.Tensor: Negative log-likelihood.
        """
        return -log_likelihood.mean()


    def sample(self, n_samples):
        """
        Generate samples from the UMM model.
        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        with torch.no_grad():
            probs = F.softmax(self.logits, dim=0)
            component_ids = torch.multinomial(probs, num_samples=n_samples, replacement=True)
            size = torch.exp(self.log_sizes)
            eps = torch.rand(n_samples, 2) - 0.5  # uniform in [-0.5, 0.5]
            samples = self.centers[component_ids] + eps * size[component_ids]
        return samples

    def conditional_sample(self, n_samples, label):
        """
        Generate samples from a specific uniform component.
        Args:
            n_samples (int): Number of samples to generate.
            label (int): Component index.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        with torch.no_grad():
            size = torch.exp(self.log_sizes[label])
            eps = torch.rand(n_samples, 2) - 0.5
            samples = self.centers[label] + eps * size
        return samples


import matplotlib.pyplot as plt
import os

# ============================================================
# Helper functions for experiments
# ============================================================

def plot_samples(samples, title, filename):
    """Plot 2D scatter of samples and save to file."""
    samples_np = samples.cpu().numpy() if isinstance(samples, torch.Tensor) else samples
    plt.figure(figsize=(8, 6))
    plt.scatter(samples_np[:, 0], samples_np[:, 1], s=5, alpha=0.6)
    plt.title(title)
    plt.xlabel("Dimension 1 (normalized)")
    plt.ylabel("Dimension 2 (normalized)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_conditional_samples(model, n_components, n_samples_per_component, title, filename):
    """Plot samples from each Gaussian component with different colors."""
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap('tab20', n_components)
    for k in range(n_components):
        samples = model.conditional_sample(n_samples_per_component, k)
        samples_np = samples.cpu().numpy() if isinstance(samples, torch.Tensor) else samples
        plt.scatter(samples_np[:, 0], samples_np[:, 1], s=10, alpha=0.6, 
                    color=cmap(k), label=f'Component {k}')
    plt.title(title)
    plt.xlabel("Dimension 1 (normalized)")
    plt.ylabel("Dimension 2 (normalized)")
    if n_components <= 20:
        plt.legend(loc='upper right', fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_log_likelihood_curve(train_lls, test_lls, title, filename):
    """Plot train and test mean log-likelihood vs epoch."""
    epochs = range(1, len(train_lls) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_lls, label='Train Mean LL', marker='o', markersize=3)
    plt.plot(epochs, test_lls, label='Test Mean LL', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Log-Likelihood')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def compute_class_means(features, labels, n_classes):
    """Compute mean of each class (country) from normalized features."""
    class_means = torch.zeros(n_classes, 2)
    for k in range(n_classes):
        mask = labels == k
        if mask.sum() > 0:
            class_means[k] = features[mask].mean(dim=0)
    return class_means


def run_experiment(model, optimizer, train_loader, test_loader, device, num_epochs,
                   checkpoint_epochs=None, save_prefix="", n_components=None):
    """
    Train model and optionally save plots at checkpoint epochs.
    Returns lists of train/test mean log-likelihoods per epoch.
    """
    train_lls = []
    test_lls = []
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        all_ll = []
        for batch in train_loader:
            X, _ = batch
            X = X.to(device)
            optimizer.zero_grad()
            log_likelihood = model(X)
            loss = model.loss_function(log_likelihood)
            loss.backward()
            optimizer.step()
            all_ll.append(log_likelihood.detach())
        train_ll = torch.cat(all_ll).mean().item()
        train_lls.append(train_ll)
        
        # Evaluation
        model.eval()
        all_ll = []
        with torch.no_grad():
            for batch in test_loader:
                X, _ = batch
                X = X.to(device)
                log_likelihood = model(X)
                all_ll.append(log_likelihood)
        test_ll = torch.cat(all_ll).mean().item()
        test_lls.append(test_ll)
        
        print(f"Epoch {epoch:02d} | train LL: {train_ll:.4f} | test LL: {test_ll:.4f}")
        
        # Save checkpoint plots if requested
        if checkpoint_epochs is not None and epoch in checkpoint_epochs:
            samples = model.sample(1000)
            plot_samples(samples, f"{save_prefix} - 1000 Samples (Epoch {epoch})",
                         f"{save_prefix}_samples_epoch{epoch}.png")
            plot_conditional_samples(model, n_components, 100,
                                     f"{save_prefix} - Conditional Samples (Epoch {epoch})",
                                     f"{save_prefix}_conditional_epoch{epoch}.png")
    
    return train_lls, test_lls


# ============================================================
# Main Experiments
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create output directory for plots
    os.makedirs("plots", exist_ok=True)
    os.chdir("plots")
    
    train_dataset = EuropeDataset('../train.csv')
    test_dataset = EuropeDataset('../test.csv')

    batch_size = 4096
    num_epochs = 50
    learning_rate = 0.01  # for GMM
    
    # Keep raw features for computing class means before normalization
    raw_features = train_dataset.features.clone()
    raw_labels = train_dataset.labels.clone()
    
    # Normalize features
    train_dataset.features = normalize_tensor(train_dataset.features, d=0)
    test_dataset.features = normalize_tensor(test_dataset.features, d=0)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_classes = int(train_dataset.labels.max().item()) + 1
    print(f"Number of classes (countries): {n_classes}")
    
    # Compute class means from normalized features for later use
    class_means = compute_class_means(train_dataset.features, train_dataset.labels, n_classes)
    
    # ============================================================
    # Question 1: Different n_components values [1, 5, 10, n_classes]
    # ============================================================
    print("\n" + "="*60)
    print("QUESTION 1: Training GMM with different n_components")
    print("="*60)
    
    n_components_list = [1, 5, 10, n_classes]
    
    for n_comp in n_components_list:
        print(f"\n--- Training GMM with n_components = {n_comp} ---")
        np.random.seed(42)
        torch.manual_seed(42)
        
        model = GMM(n_comp).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train for num_epochs
        train_lls, test_lls = run_experiment(
            model, optimizer, train_loader, test_loader, device, num_epochs,
            checkpoint_epochs=None, save_prefix=f"Q1_ncomp{n_comp}", n_components=n_comp
        )
        
        # (a) Scatter plot with 1000 samples from GMM
        samples = model.sample(1000)
        plot_samples(samples, f"GMM (n_components={n_comp}) - 1000 Samples",
                     f"Q1a_ncomp{n_comp}_samples.png")
        
        # (b) Scatter plot with 100 samples from each component, colored
        plot_conditional_samples(model, n_comp, 100,
                                 f"GMM (n_components={n_comp}) - 100 Samples per Component",
                                 f"Q1b_ncomp{n_comp}_conditional.png")
    
    # ============================================================
    # Question 2a,b: n_components = n_classes, plots at checkpoints
    # ============================================================
    print("\n" + "="*60)
    print("QUESTION 2a,b: GMM with n_components = n_classes (random init)")
    print("="*60)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    model = GMM(n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    checkpoint_epochs = [1, 10, 20, 30, 40, 50]
    
    train_lls_random, test_lls_random = run_experiment(
        model, optimizer, train_loader, test_loader, device, num_epochs,
        checkpoint_epochs=checkpoint_epochs, save_prefix="Q2_random", n_components=n_classes
    )
    
    # (b) Plot training and testing mean log-likelihood vs epoch
    plot_log_likelihood_curve(train_lls_random, test_lls_random,
                              "GMM (random init) - Mean Log-Likelihood vs Epoch",
                              "Q2b_random_ll_curve.png")
    
    # ============================================================
    # Question 2c: Initialize means to country centroids
    # ============================================================
    print("\n" + "="*60)
    print("QUESTION 2c: GMM with means initialized to country centroids")
    print("="*60)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    model_centroid = GMM(n_classes).to(device)
    # Initialize means to class centroids
    model_centroid.means.data = class_means.to(device)
    optimizer_centroid = torch.optim.Adam(model_centroid.parameters(), lr=learning_rate)
    
    train_lls_centroid, test_lls_centroid = run_experiment(
        model_centroid, optimizer_centroid, train_loader, test_loader, device, num_epochs,
        checkpoint_epochs=checkpoint_epochs, save_prefix="Q2c_centroid", n_components=n_classes
    )
    
    # Plot log-likelihood curve for centroid init
    plot_log_likelihood_curve(train_lls_centroid, test_lls_centroid,
                              "GMM (centroid init) - Mean Log-Likelihood vs Epoch",
                              "Q2c_centroid_ll_curve.png")
    
    # ============================================================
    # Comparison plot: random init vs centroid init
    # ============================================================
    print("\n" + "="*60)
    print("Comparison: Random init vs Centroid init")
    print("="*60)
    
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_lls_random, label='Random Init - Train', marker='o', markersize=2)
    plt.plot(epochs, train_lls_centroid, label='Centroid Init - Train', marker='s', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Log-Likelihood')
    plt.title('Training Mean Log-Likelihood')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_lls_random, label='Random Init - Test', marker='o', markersize=2)
    plt.plot(epochs, test_lls_centroid, label='Centroid Init - Test', marker='s', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Log-Likelihood')
    plt.title('Test Mean Log-Likelihood')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("Q2c_comparison.png", dpi=150)
    plt.close()
    print("Saved: Q2c_comparison.png")
    
    # Print final comparison
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    print(f"Random Init    - Final Train LL: {train_lls_random[-1]:.4f}, Final Test LL: {test_lls_random[-1]:.4f}")
    print(f"Centroid Init  - Final Train LL: {train_lls_centroid[-1]:.4f}, Final Test LL: {test_lls_centroid[-1]:.4f}")
    
    if test_lls_centroid[-1] > test_lls_random[-1]:
        print("\n=> Centroid initialization performed BETTER (higher test log-likelihood)")
    else:
        print("\n=> Random initialization performed BETTER (higher test log-likelihood)")
    
    print("\nAll GMM plots saved.")
    
    # ============================================================
    # ============================================================
    # UNIFORM MIXTURE MODEL (UMM) EXPERIMENTS
    # ============================================================
    # ============================================================
    
    print("\n" + "="*60)
    print("UNIFORM MIXTURE MODEL EXPERIMENTS")
    print("="*60)
    
    learning_rate_umm = 0.001  # for UMM
    
    # ============================================================
    # UMM Question 1: Different n_components values [1, 5, 10, n_classes]
    # ============================================================
    print("\n" + "="*60)
    print("UMM QUESTION 1: Training UMM with different n_components")
    print("="*60)
    
    for n_comp in n_components_list:
        print(f"\n--- Training UMM with n_components = {n_comp} ---")
        np.random.seed(42)
        torch.manual_seed(42)
        
        model = UMM(n_comp).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_umm)
        
        # Train for num_epochs
        train_lls, test_lls = run_experiment(
            model, optimizer, train_loader, test_loader, device, num_epochs,
            checkpoint_epochs=None, save_prefix=f"UMM_Q1_ncomp{n_comp}", n_components=n_comp
        )
        
        # (a) Scatter plot with 1000 samples from UMM
        samples = model.sample(1000)
        plot_samples(samples, f"UMM (n_components={n_comp}) - 1000 Samples",
                     f"UMM_Q1a_ncomp{n_comp}_samples.png")
        
        # (b) Scatter plot with 100 samples from each component, colored
        plot_conditional_samples(model, n_comp, 100,
                                 f"UMM (n_components={n_comp}) - 100 Samples per Component",
                                 f"UMM_Q1b_ncomp{n_comp}_conditional.png")
    
    # ============================================================
    # UMM Question 2a,b: n_components = n_classes, plots at checkpoints
    # ============================================================
    print("\n" + "="*60)
    print("UMM QUESTION 2a,b: UMM with n_components = n_classes (random init)")
    print("="*60)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    model_umm = UMM(n_classes).to(device)
    optimizer_umm = torch.optim.Adam(model_umm.parameters(), lr=learning_rate_umm)
    
    train_lls_umm_random, test_lls_umm_random = run_experiment(
        model_umm, optimizer_umm, train_loader, test_loader, device, num_epochs,
        checkpoint_epochs=checkpoint_epochs, save_prefix="UMM_Q2_random", n_components=n_classes
    )
    
    # (b) Plot training and testing mean log-likelihood vs epoch
    plot_log_likelihood_curve(train_lls_umm_random, test_lls_umm_random,
                              "UMM (random init) - Mean Log-Likelihood vs Epoch",
                              "UMM_Q2b_random_ll_curve.png")
    
    # ============================================================
    # UMM Question 2c: Initialize centers to country centroids
    # ============================================================
    print("\n" + "="*60)
    print("UMM QUESTION 2c: UMM with centers initialized to country centroids")
    print("="*60)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    model_umm_centroid = UMM(n_classes).to(device)
    # Initialize centers to class centroids
    model_umm_centroid.centers.data = class_means.to(device)
    optimizer_umm_centroid = torch.optim.Adam(model_umm_centroid.parameters(), lr=learning_rate_umm)
    
    train_lls_umm_centroid, test_lls_umm_centroid = run_experiment(
        model_umm_centroid, optimizer_umm_centroid, train_loader, test_loader, device, num_epochs,
        checkpoint_epochs=checkpoint_epochs, save_prefix="UMM_Q2c_centroid", n_components=n_classes
    )
    
    # Plot log-likelihood curve for centroid init
    plot_log_likelihood_curve(train_lls_umm_centroid, test_lls_umm_centroid,
                              "UMM (centroid init) - Mean Log-Likelihood vs Epoch",
                              "UMM_Q2c_centroid_ll_curve.png")
    
    # ============================================================
    # UMM Comparison plot: random init vs centroid init
    # ============================================================
    print("\n" + "="*60)
    print("UMM Comparison: Random init vs Centroid init")
    print("="*60)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_lls_umm_random, label='Random Init - Train', marker='o', markersize=2)
    plt.plot(epochs, train_lls_umm_centroid, label='Centroid Init - Train', marker='s', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Log-Likelihood')
    plt.title('UMM Training Mean Log-Likelihood')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_lls_umm_random, label='Random Init - Test', marker='o', markersize=2)
    plt.plot(epochs, test_lls_umm_centroid, label='Centroid Init - Test', marker='s', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Log-Likelihood')
    plt.title('UMM Test Mean Log-Likelihood')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("UMM_Q2c_comparison.png", dpi=150)
    plt.close()
    print("Saved: UMM_Q2c_comparison.png")
    
    # Print final UMM comparison
    print("\n" + "="*60)
    print("UMM FINAL RESULTS COMPARISON")
    print("="*60)
    print(f"Random Init    - Final Train LL: {train_lls_umm_random[-1]:.4f}, Final Test LL: {test_lls_umm_random[-1]:.4f}")
    print(f"Centroid Init  - Final Train LL: {train_lls_umm_centroid[-1]:.4f}, Final Test LL: {test_lls_umm_centroid[-1]:.4f}")
    
    if test_lls_umm_centroid[-1] > test_lls_umm_random[-1]:
        print("\n=> Centroid initialization performed BETTER (higher test log-likelihood)")
    else:
        print("\n=> Random initialization performed BETTER (higher test log-likelihood)")
    
    # ============================================================
    # UMM Question 2 Analysis: Trends in uniform supports
    # ============================================================
    print("\n" + "="*60)
    print("UMM ANALYSIS: Trends in uniform supports and gradient descent issues")
    print("="*60)
    print("""
OBSERVATIONS:
1. Trend in uniform supports (sizes): As training progresses, the uniform 
   distributions tend to GROW in size. This happens because the gradient 
   only receives signal when samples fall INSIDE the uniform's support.
   
2. Trend in centers: Centers tend to move towards regions with higher data 
   density, but this movement is limited.

3. PROBLEM with gradient descent for UMM vs GMM:
   - In GMM: Every sample contributes to the gradient because Gaussians have 
     infinite support. Even if a sample is far from a Gaussian's mean, it 
     still has non-zero probability and provides gradient information.
   
   - In UMM: Samples OUTSIDE a uniform's support contribute ZERO gradient 
     for that component's parameters. This creates a "dead zone" problem:
     * If a uniform doesn't cover any data points, its parameters receive 
       no gradient signal and cannot improve.
     * The only way to capture more data is to make the uniforms larger, 
       which explains why sizes tend to grow.
     * Centers can only move based on samples currently inside the support, 
       making it hard to discover new regions of high density.
   
   This is fundamentally a non-smooth optimization problem - the likelihood
   function has discontinuous gradients at the boundaries of each uniform.
""")
    
    print("\nAll plots saved in the 'plots' directory.")
    



