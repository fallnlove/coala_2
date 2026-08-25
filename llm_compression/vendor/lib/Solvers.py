import torch
import math
import numpy as np

def svd_low_rank_approx(W, r):
    """
    Performs a rank-r approximation of matrix W using Singular Value Decomposition (SVD).
    
    Args:
        W (torch.Tensor): The input matrix (m x d) to be approximated.
        r (int): The desired rank of the approximation (must satisfy r <= min(m, d)).
    
    Returns:
        tuple: A pair of matrices (U_r @ diag(S_r), Vh_r), whose product gives the rank-r approximation.
            - U_r @ diag(S_r): A matrix of shape (m x r)
            - Vh_r: A matrix of shape (r x d)
    
    Note:
        - Uses the "thin" (economy-sized) SVD (full_matrices=False), which computes only min(m, d) singular vectors.
        - The approximation satisfies: W ≈ (U[:, :r] @ diag(S[:r])) @ Vh[:r, :]
    """
    U, S, Vh = torch.linalg.svd(W, full_matrices=False, driver='gesvd')
    return U[:, :r] @ torch.diag(S[:r]), Vh[:r, :]


def weight_approx(W, X, r, minimize_norm=False):
    """
    Solve min_{rk W' <= r} || (W - W')X ||_F.
    
    Args:
        W (torch.Tensor): The weight matrix to approximate (m x d)
        X (torch.Tensor): Input data matrix (d x n)
        r (int): Rank of the approximation
        minimize_norm (bool, optional): If True, minimizes the norm of the approximation. 
                                    Defaults to False.
    
    Returns:
        tuple: A pair of tensors (A, B) where A is (m x r) and B is (r x d)    
    Note:
        - If X has fewer rows than columns (d < n), uses QR decomposition for speed
        - The approximation is computed using SVD on W @ S where S is either X or its QR factor
        - When minimize_norm=True, projects the solution onto the column space of X
    """
    if X.shape[0] <= X.shape[1]:
        _, S = torch.linalg.qr(X.T, mode='r')
        U, _, _ = torch.linalg.svd(W @ S.T, full_matrices=False, driver='gesvd')
    else:
        U, _, _ = torch.linalg.svd(W @ X, full_matrices=False, driver='gesvd')
    U = U[:, :r]
    A = U
    B = U.T @ W
    if minimize_norm:
        Q, _ = torch.linalg.qr(X)
        B = (B @ Q) @ Q.T
    return A, B


def weight_approx_with_regularization(W, X, r, mu):
    """
    Solve min_{rk W' <= r} || (W - W')X ||^2_F + mu * || W - W' ||^2_F
    
    Args:
        W (torch.Tensor): The weight matrix to approximate (m x d)
        X (torch.Tensor): Input data matrix (d x n)
        r (int): Rank of the approximation
        mu (float): Regularization parameter
    
    Returns:
        tuple: A pair of tensors (A, B) where A is (m x r) and B is (r x d)
    
    Note:
        - Adds a scaled identity matrix to X for regularization
        - Calls weight_approx on the augmented matrix [X, sqrt(mu)*I]
        - Helps prevent overfitting and improves numerical stability
    """
    I = math.sqrt(mu) * torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    X_new = torch.cat([X, I], dim=1)
    return weight_approx(W, X_new, r)


def svd_llm_method(W, X, r, eps=0):
    """
    Implements the SVD-LLM method for efficient LLM weight matrix approximation.
    Computes a rank-r approximation of weight matrix W using input data statistics.
    
    Args:
        W (torch.Tensor): The weight matrix to approximate (shape m x n)
        X (torch.Tensor): Input data matrix (shape d x N) where:
                        - d is the input dimension
                        - N is the number of data samples
        r (int): Rank of the desired approximation
        eps (float): Small regularization term for numerical stability (default: 0)
    
    Returns:
        tuple: (U_r, V_r) where:
            - U_r is the left singular vectors (shape m x r)
            - V_r is the projected right singular vectors (shape r x n)
    
    Note:
        - Based on "SVD-LLM: Efficient Compression of Large Language Models via Singular Value Decomposition"
        - Paper available at https://arxiv.org/abs/2403.07378
        - The method performs SVD on a whitened version of W using input covariance statistics
    """
    Cov = X @ X.T
    if not torch.isfinite(Cov).all():
        Cov = torch.nan_to_num(Cov, nan=0.0, posinf=1.0, neginf=-1.0)
    
    while True:
        try:
            S = torch.linalg.cholesky(Cov + eps * torch.eye(Cov.shape[0], device=X.device))
            break
        except Exception as e:
            if eps == 0:
                eps = 1e-9
            else:
                eps *= 2
    U, Sigma, Vt = torch.linalg.svd(W @ S, full_matrices=False)
    S_inv = torch.linalg.inv(S)
    return U[:, :r] @ torch.diag(Sigma[:r]), Vt[:r, :] @ S_inv


def svd_llm_2_method(W, X, r):
    """
    Implements the SVD-LLM V2 method from arXiv:2503.12340 for efficient weight matrix approximation.
    
    Args:
        W (torch.Tensor): The weight matrix to approximate (shape: output_dim x input_dim)
        X (torch.Tensor): Input data matrix (shape: input_dim x num_samples)
        r (int): Rank of the approximation
    
    
    Note:
        - Reference: "SVD-LLM V2: Optimizing Singular Value Truncation for Large Language Model Compression"
        - Paper available at https://arxiv.org/abs/2503.12340
    """
    Cov = X @ X.T
    U_s, S_s, _ = torch.linalg.svd(Cov, full_matrices=False)
    D = W @ U_s @ torch.diag(torch.sqrt(S_s))
    U, S, Vt = torch.linalg.svd(D, full_matrices=False)
    return U[:, :r], torch.diag(S[:r]) @ Vt[:r, :] @ torch.diag(1 / torch.sqrt(S_s)) @ U_s.T
    

def swift_svd_project(W, X):
    """Z = W @ whiten(X): the matrix whose LEFT singular vectors are the right
    singular vectors V of Y = X_calib Wᵀ, with the same singular values.

    Shared by ``swift_svd_method`` and ``pipeline/swift_rank.py`` so that the
    spectrum driving the rank allocation, the cached projectors used to score
    candidates, and the factors finally injected all agree numerically.

    The QR pre-step mirrors ``weight_approx``: X Xᵀ is unchanged (COALA's hook
    already stores X with X Xᵀ = sum_batch Xᵀ X exactly), only the column count
    shrinks.
    """
    # Finite guard (same rationale as asvd_method): half-precision calibration
    # on outlier-heavy models can leave inf/nan in the cached X, which would
    # poison the SVD and hence every factor of this layer.
    if not torch.isfinite(X).all():
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if X.shape[0] <= X.shape[1]:
        _, S = torch.linalg.qr(X.T, mode='r')
        return W @ S.T
    return W @ X


def swift_svd_method(W, X, r):
    """Swift-SVD (arXiv:2604.01609, ICML 2026): out-side spectral projection.

    Theorem 3.1: with Y = X_calib Wᵀ the layer's output activations, the rank-r
    weight minimising ||X_calib Wᵀ - X_calib W'ᵀ||_F is W' = V_r V_rᵀ W where
    V_r are the top-r RIGHT singular vectors of Y, and the residual is exactly
    the singular tail sqrt(sum_{j>r} sigma_j^2). The authors obtain V by an
    eigendecomposition of the aggregated covariance YᵀY.

    Here the equivalent (cheaper, better-conditioned) route is used: COALA's
    hook stores the compacted INPUT activations X with X Xᵀ = sum_batch Xᵀ X
    exactly (QR compaction, see coala/Injector.py:compactify), so
    Y ᵀY = (W X)(W X)ᵀ and V = left singular vectors of Z = W @ X. The QR
    pre-step mirrors weight_approx: same Gram, fewer columns.

    Returns (B, A) with W' = B @ A: B = V_r (m x r), A = V_rᵀ W (r x n), so
    COALA_Linear stores adapter_A = Wᵀ V_r and adapter_B = V_rᵀ — the factored
    form of the paper (Fig. 1: cache X A_k instead of X Wᵀ).

    Note: this closed form coincides with COALA's unregularised operator
    (weight_approx, mu=0). Uniform Swift-SVD is therefore expected to reproduce
    COALA(mu=0) numerically; the methods differ in rank ALLOCATION
    (pipeline/swift_rank.py implements the paper's dynamic allocation).
    """
    M = swift_svd_project(W, X)
    driver = 'gesvd' if W.is_cuda else None  # cuSOLVER driver: CUDA-only
    U, _, _ = torch.linalg.svd(M, full_matrices=False, driver=driver)
    V_r = U[:, :r]
    return V_r, V_r.T @ W


def asvd_method(W, X, r, alpha):
    """
    Implementation of the ASVD method described in arXiv:2312.05821.
    
    Args:
        W (torch.Tensor): The weight matrix to approximate
        X (torch.tensor): Input data matrix
        r (int): Rank of the approximation
        alpha (float): Scaling factor for the activation-aware scaling
    
    Returns:
        tuple: A tuple containing:
            - U (torch.Tensor): Left singular vectors of the approximation
            - SV (torch.Tensor): Product of singular values and right singular vectors, scaled by inverse activation statistics
    
    Note:
        - Reference: "ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models"
        - Paper available at https://arxiv.org/abs/2312.05821
    """
    # Finite guard (mirrors svd_llm_method's Cov sanitisation): half-precision
    # calibration on models with large activation outliers (e.g. Qwen3) can put
    # inf into the cached X (fp16 overflow) or underflow whole channels to 0.
    # Unguarded, inf poisons the SVD (NaN factors) and mean_abs=0 makes the
    # trailing diag(1/S) infinite. Drop non-finite entries and floor S.
    if not torch.isfinite(X).all():
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mean_abs = torch.mean(torch.abs(X), dim=1)
    S = (mean_abs ** alpha).clamp_min(1e-6)
    U, Sigma, Vt = torch.linalg.svd(W @ torch.diag(S), full_matrices=False)

    return U[:, :r], torch.diag(Sigma[:r]) @ Vt[:r, :] @ torch.diag(1 / S)