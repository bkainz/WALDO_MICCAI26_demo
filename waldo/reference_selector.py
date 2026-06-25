"""
Entropy-weighted Sliced Wasserstein reference selection.

This module implements WALDO's reference-selection stage exactly as described in
the paper (Sec. "Method"):

  1. Extract DINOv3-ViT-B/16 patch tokens  phi_1..phi_T  in R^768.
  2. Weight each patch by its softmax-Shannon entropy
        w_i = H(phi_i) / sum_j H(phi_j),   H(phi) = -sum_k sigma_k(phi) log sigma_k(phi)
     where sigma(.) is the softmax over feature dimensions.
  3. Score every healthy reference by the entropy-weighted, squared
     Sliced Wasserstein distance to the query
        SW2^(w)(P, Q) = ( sum_i w_i (p_(i) - q_(i))^2 )^(1/2)
     averaged over M = 100 random unit projections (sorted 1-D transport).
  4. Keep references whose distance falls in the "Goldilocks zone"
     (30th-70th percentile, i.e. alpha = 0.3).
  5. Select K diverse references from that zone with a Determinantal Point
     Process (greedy log-det MAP) using the kernel
        L_ij = q_i q_j exp(-beta * SW2(h_i, h_j)),  q_i = 1[h_i in Goldilocks zone].

NOTE (distillation disclaimer): this is a clean reference implementation distilled
by Claude and Codex coding agents from a larger internal experimentation codebase.
It faithfully implements the method and hyper-parameters reported in the paper; it
is not the exact orchestration script used to produce the published tables (which
ran across a multi-day cluster job). See DISCLAIMER.md.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

# Paper hyper-parameters (Sec. "Implementation").
DINOV3_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # ViT-B/16, 768-d patch tokens
DINOV2_FALLBACK = "facebook/dinov2-base"                    # ungated fallback if DINOv3 is unavailable
N_PROJECTIONS = 100        # M random projections for Sliced Wasserstein
FEATURE_SIZE = 512         # images resized to 512x512 for feature extraction
GOLDILOCKS_PCT = (30, 70)  # percentile band -> alpha = 0.3
DPP_BETA = 1.0             # diversity sharpness in the DPP kernel


class WassersteinReferenceSelector:
    """Select healthy references via entropy-weighted Sliced Wasserstein + DPP.

    Args:
        model_name: HuggingFace id of the DINOv3 backbone (ViT-B/16). DINOv3 is gated
            on the Hub (requires accepting the licence + an auth token). If it cannot
            be loaded the constructor RAISES with setup instructions, unless
            allow_dinov2_fallback=True (see below).
        device: torch device ("cuda" or "cpu").
        n_projections: number of random projections M for SW approximation.
        feature_size: square resize applied before feature extraction.
        goldilocks_percentiles: (low, high) percentile band defining the Goldilocks zone.
        dpp_beta: sharpness of the DPP diversity kernel.
        seed: RNG seed for the random projections (the paper uses seed 42).
        allow_dinov2_fallback: if True, fall back to the (non-paper) DINOv2-base
            backbone when DINOv3 is unavailable, instead of raising. Default False.
    """

    def __init__(
        self,
        model_name: str = DINOV3_MODEL,
        device: str = "cuda",
        n_projections: int = N_PROJECTIONS,
        feature_size: int = FEATURE_SIZE,
        goldilocks_percentiles: Tuple[int, int] = GOLDILOCKS_PCT,
        dpp_beta: float = DPP_BETA,
        seed: Optional[int] = 42,
        allow_dinov2_fallback: bool = False,
    ):
        self.device = device
        self.n_projections = n_projections
        self.feature_size = feature_size
        self.goldilocks_percentiles = goldilocks_percentiles
        self.dpp_beta = dpp_beta
        self._rng = np.random.default_rng(seed)
        self.model_name = model_name
        self.allow_dinov2_fallback = allow_dinov2_fallback
        self._load_model(model_name)

    # ------------------------------------------------------------------ #
    # Backbone
    # ------------------------------------------------------------------ #
    def _load_model(self, model_name: str):
        """Load the DINOv3-ViT-B/16 backbone (the paper backbone).

        DINOv3 is gated on the HuggingFace Hub. If it cannot be loaded we do NOT
        silently substitute a different backbone: by default we raise with clear
        instructions, so that running the demo always uses the paper's backbone.
        Pass ``allow_dinov2_fallback=True`` to explicitly opt in to the
        (non-paper) DINOv2-base backbone for a quick smoke run.
        """
        from transformers import AutoImageProcessor, AutoModel

        self.is_dinov3 = "dinov3" in model_name.lower()
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except Exception as exc:  # gated model / no token / offline
            if model_name == DINOV2_FALLBACK:
                raise
            if not self.allow_dinov2_fallback:
                raise RuntimeError(
                    f"Could not load the paper backbone DINOv3-ViT-B/16 "
                    f"('{model_name}'): {exc}\n"
                    f"DINOv3 is gated on the HuggingFace Hub. Accept its licence at "
                    f"https://huggingface.co/{model_name} and run `huggingface-cli login`.\n"
                    f"To run a quick (non-paper) smoke test with DINOv2-base instead, "
                    f"set allow_dinov2_fallback=True (or pass --allow-dinov2-fallback)."
                ) from exc
            print(
                f"[WALDO] WARNING: DINOv3 backbone '{model_name}' unavailable ({exc}).\n"
                f"        Falling back to NON-PAPER backbone '{DINOV2_FALLBACK}' because "
                f"allow_dinov2_fallback=True. Results will NOT match the paper."
            )
            self.is_dinov3 = False
            self.model_name = DINOV2_FALLBACK
            self.processor = AutoImageProcessor.from_pretrained(DINOV2_FALLBACK)
            self.model = AutoModel.from_pretrained(DINOV2_FALLBACK)
        self.model = self.model.to(self.device)
        self.model.eval()
        # Number of non-patch prefix tokens: 1 CLS + register tokens (DINOv3 has 4).
        self.num_register_tokens = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        self.num_prefix_tokens = 1 + self.num_register_tokens

    def extract_patches(self, image: np.ndarray) -> np.ndarray:
        """Return DINOv3 patch tokens of shape (T, D) for an RGB image (H, W, 3)."""
        import torch

        inputs = self.processor(
            images=image,
            return_tensors="pt",
            size={"height": self.feature_size, "width": self.feature_size},
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        tokens = outputs.last_hidden_state[0]  # (1 + reg + T, D)
        patches = tokens[self.num_prefix_tokens:, :]  # drop CLS + registers -> (T, D)
        return patches.float().cpu().numpy()

    # ------------------------------------------------------------------ #
    # Entropy weighting  (paper Eq. for w_i, H(phi))
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_entropy_weights(patches: np.ndarray) -> np.ndarray:
        """Softmax-Shannon entropy weights over feature dimensions, normalised to sum 1.

        H(phi) = -sum_k sigma_k(phi) log sigma_k(phi),  sigma = softmax over feature dims.
        w_i = H(phi_i) / sum_j H(phi_j).  High-entropy (textured) patches get more weight.
        """
        x = patches - patches.max(axis=1, keepdims=True)  # numerical stability
        ex = np.exp(x)
        sigma = ex / (ex.sum(axis=1, keepdims=True) + 1e-12)  # softmax over feature dims
        entropy = -np.sum(sigma * np.log(sigma + 1e-12), axis=1)  # (T,)
        total = entropy.sum()
        if total <= 0:
            return np.full(patches.shape[0], 1.0 / patches.shape[0])
        return entropy / total

    # ------------------------------------------------------------------ #
    # Entropy-weighted, squared Sliced Wasserstein  (paper Eq. 1)
    # ------------------------------------------------------------------ #
    def sliced_wasserstein_distance(
        self,
        patches1: np.ndarray,
        patches2: np.ndarray,
        weights1: Optional[np.ndarray] = None,
        weights2: Optional[np.ndarray] = None,  # kept for API symmetry; query weights drive Eq. 1
    ) -> float:
        """Entropy-weighted squared Sliced Wasserstein distance SW2^(w)(patches1, patches2).

        For each random unit projection theta the 1-D 2-Wasserstein between equal-size
        empirical distributions reduces to sorted differences; weights follow the
        (sorted) patches of ``patches1`` per the paper's Eq. 1:
            SW2^(w) = ( mean_theta  sum_i w_(i) (p_(i) - q_(i))^2 )^(1/2).
        """
        d = patches1.shape[1]
        if weights1 is None:
            weights1 = np.full(patches1.shape[0], 1.0 / patches1.shape[0])

        theta = self._rng.standard_normal((self.n_projections, d))
        theta /= (np.linalg.norm(theta, axis=1, keepdims=True) + 1e-12)

        proj1 = patches1 @ theta.T  # (T1, M)
        proj2 = patches2 @ theta.T  # (T2, M)

        order1 = np.argsort(proj1, axis=0)
        order2 = np.argsort(proj2, axis=0)
        p_sorted = np.take_along_axis(proj1, order1, axis=0)
        q_sorted = np.take_along_axis(proj2, order2, axis=0)
        # weights of patches1 in sorted order, per projection
        w_sorted = np.take_along_axis(np.broadcast_to(weights1[:, None], proj1.shape), order1, axis=0)

        t1, t2 = p_sorted.shape[0], q_sorted.shape[0]
        if t1 != t2:  # resample q's quantiles onto p's grid (rare: only if sizes differ)
            grid = np.linspace(0, 1, t1)
            src = np.linspace(0, 1, t2)
            q_sorted = np.stack(
                [np.interp(grid, src, q_sorted[:, m]) for m in range(self.n_projections)], axis=1
            )

        per_proj = np.sum(w_sorted * (p_sorted - q_sorted) ** 2, axis=0)  # (M,)
        return float(np.sqrt(np.mean(per_proj)))

    # ------------------------------------------------------------------ #
    # DPP diversity (greedy log-det MAP)
    # ------------------------------------------------------------------ #
    def _dpp_greedy(self, sw_pairwise: np.ndarray, k: int) -> List[int]:
        """Greedy MAP selection of k items maximising log det(L_S).

        L_ij = exp(-beta * SW2(h_i, h_j))  (q_i = 1 inside the Goldilocks zone).
        """
        n = sw_pairwise.shape[0]
        k = min(k, n)
        L = np.exp(-self.dpp_beta * sw_pairwise)
        L = L + 1e-6 * np.eye(n)
        selected: List[int] = []
        remaining = list(range(n))
        for _ in range(k):
            best_idx, best_gain = None, -np.inf
            for idx in remaining:
                cand = selected + [idx]
                sign, logdet = np.linalg.slogdet(L[np.ix_(cand, cand)])
                gain = logdet if sign > 0 else -np.inf
                if gain > best_gain:
                    best_gain, best_idx = gain, idx
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
        return selected

    # ------------------------------------------------------------------ #
    # Public selection API
    # ------------------------------------------------------------------ #
    def _select(
        self,
        query_image: np.ndarray,
        reference_images: Sequence[np.ndarray],
        n_references: int,
    ) -> List[Tuple[int, float]]:
        """Core selection. Returns list of (reference_index, SW-distance-to-query)."""
        query_patches = self.extract_patches(query_image)
        query_weights = self.compute_entropy_weights(query_patches)

        ref_patches = [self.extract_patches(img) for img in reference_images]
        distances = np.array(
            [self.sliced_wasserstein_distance(query_patches, rp, query_weights) for rp in ref_patches]
        )

        # Goldilocks zone: distances within the [low, high] percentile band.
        lo = np.percentile(distances, self.goldilocks_percentiles[0])
        hi = np.percentile(distances, self.goldilocks_percentiles[1])
        zone = np.where((distances >= lo) & (distances <= hi))[0]
        if len(zone) < n_references:  # fall back to references closest to the median
            zone = np.argsort(np.abs(distances - np.median(distances)))[: max(n_references, 1)]

        if len(zone) <= n_references:
            chosen = list(zone)
        else:
            # DPP diversity over the zone using pairwise SW between candidate references.
            m = len(zone)
            sw_pair = np.zeros((m, m))
            zone_weights = [self.compute_entropy_weights(ref_patches[g]) for g in zone]
            for a in range(m):
                for b in range(a + 1, m):
                    dist = self.sliced_wasserstein_distance(
                        ref_patches[zone[a]], ref_patches[zone[b]], zone_weights[a]
                    )
                    sw_pair[a, b] = sw_pair[b, a] = dist
            picked = self._dpp_greedy(sw_pair, n_references)
            chosen = [int(zone[i]) for i in picked]

        return [(int(i), float(distances[i])) for i in chosen]

    def select_references(
        self,
        query_image: np.ndarray,
        reference_images: Sequence[np.ndarray],
        n_references: int = 5,
    ) -> List[int]:
        """Return indices of the K selected Goldilocks-zone references (DPP-diverse)."""
        return [idx for idx, _ in self._select(query_image, reference_images, n_references)]

    def select_references_with_scores(
        self,
        query_image: np.ndarray,
        reference_images: Sequence[np.ndarray],
        n_references: int = 5,
    ) -> List[Tuple[int, float]]:
        """Return (index, SW-distance-to-query) for each selected reference.

        The distance is used downstream to weight VLM predictions via
        c~ = c * exp(-lambda * SW2(P_q, P_h)).
        """
        return self._select(query_image, reference_images, n_references)
