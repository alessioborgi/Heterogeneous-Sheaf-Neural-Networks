"""
Lazy Laplacian Builders
=======================y
- Use apply(H) to compute L_F @ H without ever forming L_F
- Memory: O(m * d^2) instead of O(m * d^2 + n * d^2) for general sheaves
"""

import torch
from torch import nn
from torch_geometric.utils import degree
from torch_scatter import scatter_add
from .orthogonal import Orthogonal


class LazyLaplacianBuilder(nn.Module):
    """
    Base class for lazy Laplacian builders.

    Instead of constructing L_F explicitly, stores restriction maps and
    computes L_F @ H on demand via the coboundary factorization:
        L_F = B_F^T @ B_F
    where B_F is the weighted coboundary operator.
    """

    def __init__(
        self,
        size,
        edge_index,
        d,
        normalised=False,
        deg_normalised=False,
        add_hp=False,
        add_lp=False,
        augmented=True,
    ):
        super(LazyLaplacianBuilder, self).__init__()
        assert not (normalised and deg_normalised)

        self.d = d
        self.final_d = d
        if add_hp:
            self.final_d += 1
        if add_lp:
            self.final_d += 1
        self.size = size
        self.normalised = normalised
        self.deg_normalised = deg_normalised
        self.add_hp = add_hp
        self.add_lp = add_lp
        self.augmented = augmented

        # Edge-dependent buffers
        self.edge_index = None
        self.edges = 0
        self.device = None
        self.deg = None

        # Cached normalization for apply()
        self._D_inv_sqrt = None

        self._reset_edge_dependent_buffers(edge_index)

    def _post_edge_index_update(self):
        """Hook for subclasses to recompute cached tensors."""
        pass

    def _reset_edge_dependent_buffers(self, edge_index):
        self.edge_index = edge_index
        if edge_index is None:
            self.edges = 0
            self.device = None
            self.deg = None
            self._D_inv_sqrt = None
            self._post_edge_index_update()
            return

        assert edge_index.size(0) == 2
        self.edges = edge_index.size(1) // 2
        self.device = edge_index.device
        self.deg = degree(edge_index[0], num_nodes=self.size)

        self._post_edge_index_update()

    def update_graph(self, edge_index, size=None):
        """Update for new graph topology (used in graph classification)."""
        if size is not None and size != self.size:
            self.size = size
        if edge_index is None:
            raise ValueError("edge_index must be provided when updating the Laplacian.")
        if edge_index.numel() > 0 and edge_index.max() >= self.size:
            raise ValueError(
                f"edge_index contains node id {int(edge_index.max())} >= size {self.size}"
            )
        self._reset_edge_dependent_buffers(edge_index)
        return self

    def apply(self, H):
        """
        Compute L_F @ H without constructing L_F.

        Args:
            H: Node features [n, d] or [n * d] (flattened)

        Returns:
            L_F @ H with same shape as input
        """
        raise NotImplementedError("Subclasses must implement apply()")

    def create_with_new_edge_index(self, edge_index):
        """Create a new builder with different edge_index.

        Subclasses with extra constructor parameters should override this.
        """
        assert edge_index.max() <= self.size
        new_builder = self.__class__(
            self.size,
            edge_index,
            self.d,
            normalised=self.normalised,
            deg_normalised=self.deg_normalised,
            add_hp=self.add_hp,
            add_lp=self.add_lp,
            augmented=self.augmented,
        )
        new_builder.train(self.training)
        return new_builder

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _reshape_input(self, H):
        """
        Normalise feature tensor H into [n, final_d, c] and record how to
        restore the original layout.

        Returns:
            H_3d:  [n, final_d, c]
            meta:  dict consumed by ``_restore_output``
        """
        input_shape = H.shape
        meta = {"input_shape": input_shape, "squeeze_last": False, "flatten": False, "pack": False}

        if H.dim() == 2 and H.size(0) == self.size * self.final_d:
            H = H.contiguous().view(self.size, self.final_d, H.size(1))
            meta["flatten"] = True
        elif H.dim() == 2 and H.size(0) == self.size and H.size(1) == self.final_d:
            H = H.unsqueeze(-1)
            meta["squeeze_last"] = True
        elif H.dim() == 2 and H.size(0) == self.size and (H.size(1) % self.final_d) == 0:
            H = H.contiguous().view(self.size, self.final_d, -1)
            meta["pack"] = True
        elif H.dim() == 2 and H.size(0) == self.size * self.d and self.final_d == self.d:
            H = H.contiguous().view(self.size, self.d, H.size(1))
            meta["flatten"] = True
        elif H.dim() == 3 and H.size(0) == self.size and H.size(1) == self.final_d:
            pass  # already [n, final_d, c]
        else:
            raise ValueError(
                f"Unexpected input shape: {input_shape}. Expected one of: "
                f"[n*final_d, c], [n, final_d*c], [n, final_d], or [n, final_d, c] "
                f"with final_d={self.final_d}."
            )

        return H, meta

    @staticmethod
    def _restore_output(out, meta):
        """Restore the output tensor to the caller's original layout."""
        if meta["squeeze_last"]:
            out = out.squeeze(-1)
        if meta["flatten"] or meta["pack"]:
            out = out.view(meta["input_shape"])
        return out

    def _apply_fixed_channels(self, H_fixed, row, col):
        """
        Compute the fixed LP / HP channel contributions.

        Args:
            H_fixed: [n, num_fixed, c]   (may have 0 elements)
            row, col: edge index vectors

        Returns:
            out_fixed: [n, num_fixed, c] or None
        """
        if H_fixed.numel() == 0:
            return None

        if self.normalised or self.deg_normalised:
            deg = self.deg
            if self.augmented:
                deg_inv_sqrt = (deg + 1).pow(-0.5)
            else:
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        else:
            deg_inv_sqrt = None

        def _scalar_op(h_scalar: torch.Tensor, offdiag_sign: float) -> torch.Tensor:
            """Apply (D ± A) with optional symmetric normalization."""
            h_in = deg_inv_sqrt.unsqueeze(-1) * h_scalar if deg_inv_sqrt is not None else h_scalar
            h_src_s = h_in[row]
            h_dst_s = h_in[col]
            delta_s = h_src_s + offdiag_sign * h_dst_s
            out_s = torch.zeros_like(h_in)
            out_s.scatter_add_(0, row.view(-1, 1).expand_as(delta_s), delta_s)
            if deg_inv_sqrt is not None:
                out_s = deg_inv_sqrt.unsqueeze(-1) * out_s
            return out_s

        chunks = []
        idx = 0
        if self.add_lp:
            chunks.append(_scalar_op(H_fixed[:, idx, :], offdiag_sign=+1.0).unsqueeze(1))
            idx += 1
        if self.add_hp:
            chunks.append(_scalar_op(H_fixed[:, idx, :], offdiag_sign=-1.0).unsqueeze(1))
            idx += 1

        return torch.cat(chunks, dim=1) if chunks else None


class LazyDiagLaplacianBuilder(LazyLaplacianBuilder):
    """
    Lazy Sheaf Laplacian with diagonal restriction maps.

    Restriction maps F_v▷e are diagonal matrices represented as vectors f ∈ R^d.

    The Laplacian action L_F @ H is computed as:
        (L_F @ H)_v = Σ_{e=(v,u)} f_{v▷e} ⊙ (f_{v▷e} ⊙ h_v − f_{u▷e} ⊙ h_u)

    where ⊙ is element-wise multiplication.
    """

    def __init__(
        self,
        size,
        edge_index,
        d,
        normalised=False,
        deg_normalised=False,
        add_hp=False,
        add_lp=False,
        augmented=True,
    ):
        super(LazyDiagLaplacianBuilder, self).__init__(
            size, edge_index, d, normalised, deg_normalised, add_hp, add_lp, augmented
        )
        self._maps = None
        self._diag_sum = None  # Σ_e f_v² per node, for normalization

    def forward(self, maps):
        """
        Process and store restriction maps.

        Args:
            maps: Diagonal restriction maps [m, d] where m = number of directed edges

        Returns:
            (maps, saved_maps)
        """
        assert maps.dim() == 2
        assert maps.size(1) == self.d

        row, _ = self.edge_index

        self._maps = maps

        # Diagonal of the block-diagonal degree matrix: D_vv = diag(Σ_e f_{v▷e}²)
        self._diag_sum = scatter_add(maps ** 2, row, dim=0, dim_size=self.size)

        # Precompute normalization
        if self.normalised:
            if self.augmented:
                self._D_inv_sqrt = (self._diag_sum + 1).pow(-0.5)
            else:
                self._D_inv_sqrt = self._diag_sum.pow(-0.5)
                self._D_inv_sqrt.masked_fill_(self._D_inv_sqrt == float("inf"), 0)
        elif self.deg_normalised:
            deg_sqrt_inv = (self.deg + 1).pow(-0.5) if self.augmented else self.deg.pow(-0.5)
            deg_sqrt_inv.masked_fill_(deg_sqrt_inv == float("inf"), 0)
            self._D_inv_sqrt = deg_sqrt_inv.unsqueeze(-1).expand(-1, self.d)
        else:
            self._D_inv_sqrt = None

        saved_maps = maps.detach().clone()
        return maps, saved_maps

    def apply(self, H):
        """
        Compute L_F @ H lazily for diagonal sheaves.

        If normalization is enabled the result is D^{-1/2} L_unnorm D^{-1/2} H.
        """
        if self._maps is None:
            raise RuntimeError("Must call forward() with maps before apply()")

        H, meta = self._reshape_input(H)

        row, col = self.edge_index
        f = self._maps  # [m, d]

        # Split learnable / fixed channels
        H_learn = H[:, : self.d, :]  # [n, d, c]
        H_fixed = H[:, self.d :, :]  # [n, num_fixed, c]

        # --- Learnable sheaf part: D^{-1/2} L_unnorm D^{-1/2} H_learn ---
        # Pre-scale by D^{-1/2} when normalised
        if self._D_inv_sqrt is not None:
            H_learn = self._D_inv_sqrt.unsqueeze(-1) * H_learn

        h_src = H_learn[row]  # [m, d, c]
        h_dst = H_learn[col]  # [m, d, c]

        delta = f.unsqueeze(-1) * h_src - f.unsqueeze(-1) * h_dst  # [m, d, c]
        contrib = f.unsqueeze(-1) * delta  # [m, d, c]

        out_learn = torch.zeros_like(H_learn)
        out_learn.scatter_add_(0, row.view(-1, 1, 1).expand_as(contrib), contrib)

        # Post-scale by D^{-1/2}
        if self._D_inv_sqrt is not None:
            out_learn = self._D_inv_sqrt.unsqueeze(-1) * out_learn

        # --- Fixed LP / HP channels ---
        out_fixed = self._apply_fixed_channels(H_fixed, row, col)

        # Recombine
        out = out_learn if out_fixed is None else torch.cat([out_learn, out_fixed], dim=1)
        return self._restore_output(out, meta)


class LazyGeneralLaplacianBuilder(LazyLaplacianBuilder):
    """
    Lazy Sheaf Laplacian with general d×d restriction maps.

    The Laplacian action L_F @ H is computed via coboundary factorization:
        δ_e  = F_{src▷e} h_src − F_{dst▷e} h_dst   (coboundary)
        (L_F H)_v = Σ_{e: v∈e} F_{v▷e}^T δ_e       (adjoint)
    """

    def __init__(
        self,
        size,
        edge_index,
        d,
        normalised=False,
        deg_normalised=False,
        add_hp=False,
        add_lp=False,
        augmented=True,
    ):
        super(LazyGeneralLaplacianBuilder, self).__init__(
            size, edge_index, d, normalised, deg_normalised, add_hp, add_lp, augmented
        )
        self._maps = None
        self._diag_blocks = None  # [n, d, d] block-diagonal degree

    def forward(self, maps):
        """
        Process and store restriction maps.

        Args:
            maps: General restriction maps [m, d, d]

        Returns:
            (maps, saved_maps)
        """
        assert maps.dim() == 3
        assert maps.size(1) == self.d and maps.size(2) == self.d

        row, _ = self.edge_index

        # Sanitise non-finite entries
        if not torch.all(torch.isfinite(maps)):
            num_bad = (~torch.isfinite(maps)).sum().item()
            print(f"Warning: {num_bad} non-finite values in maps. Replacing with zeros.")
            maps = torch.where(torch.isfinite(maps), maps, torch.zeros_like(maps))

        self._maps = maps

        # Block-diagonal degree: D_v = Σ_{e: v∈e} F_{v▷e}^T F_{v▷e}
        FtF = torch.bmm(maps.transpose(-1, -2), maps)  # [m, d, d]
        self._diag_blocks = scatter_add(FtF, row, dim=0, dim_size=self.size)  # [n, d, d]

        # Normalization
        if self.normalised:
            self._compute_block_normalization()
        elif self.deg_normalised:
            s = (self.deg * self.d + 1) if self.augmented else (self.deg * self.d)
            deg_sqrt_inv = s.pow(-0.5)
            deg_sqrt_inv.masked_fill_(deg_sqrt_inv == float("inf"), 0)
            self._D_inv_sqrt = deg_sqrt_inv.view(-1, 1, 1)  # scalar per node
        else:
            self._D_inv_sqrt = None

        saved_maps = maps.detach().clone()
        return maps, saved_maps

    def _compute_block_normalization(self):
        """Compute D^{-1/2} for block-diagonal normalization via eigendecomposition."""
        D = self._diag_blocks  # [n, d, d]

        if self.augmented:
            eye = torch.eye(self.d, device=D.device, dtype=D.dtype).unsqueeze(0)
            D = D + eye

        try:
            # Small deterministic epsilon for numerical stability
            eps = 1e-6 * torch.eye(self.d, device=D.device, dtype=D.dtype).unsqueeze(0)
            D = D + eps

            eigenvalues, eigenvectors = torch.linalg.eigh(D)
            eigenvalues = eigenvalues.clamp(min=1e-8)
            D_inv_sqrt = (
                eigenvectors
                @ torch.diag_embed(eigenvalues.pow(-0.5))
                @ eigenvectors.transpose(-1, -2)
            )
            self._D_inv_sqrt = D_inv_sqrt  # [n, d, d]
        except Exception as e:
            print(f"Warning: Block normalization failed ({e}), falling back to identity")
            self._D_inv_sqrt = (
                torch.eye(self.d, device=D.device, dtype=D.dtype)
                .unsqueeze(0)
                .expand(self.size, -1, -1)
            )

    def apply(self, H):
        """
        Compute L_F @ H lazily for general sheaves.

        If normalization is enabled the result is D^{-1/2} L_unnorm D^{-1/2} H.
        """
        if self._maps is None:
            raise RuntimeError("Must call forward() with maps before apply()")

        H, meta = self._reshape_input(H)

        row, col = self.edge_index
        F = self._maps  # [m, d, d]

        H_learn = H[:, : self.d, :]  # [n, d, c]
        H_fixed = H[:, self.d :, :]

        # Pre-scale learnable input
        if self._D_inv_sqrt is not None:
            if self._D_inv_sqrt.dim() == 3 and self._D_inv_sqrt.size(-1) == self.d:
                H_learn = torch.bmm(self._D_inv_sqrt, H_learn)
            else:
                H_learn = self._D_inv_sqrt * H_learn

        # Gather
        h_src = H_learn[row]  # [m, d, c]
        h_dst = H_learn[col]

        # Coboundary
        Fh_src = torch.bmm(F, h_src)
        Fh_dst = torch.bmm(F, h_dst)
        delta = Fh_src - Fh_dst

        # Adjoint
        Ft_delta = torch.bmm(F.transpose(-1, -2), delta)

        # Scatter
        out_learn = torch.zeros_like(H_learn)
        out_learn.scatter_add_(0, row.view(-1, 1, 1).expand_as(Ft_delta), Ft_delta)

        # Post-scale learnable output
        if self._D_inv_sqrt is not None:
            if self._D_inv_sqrt.dim() == 3 and self._D_inv_sqrt.size(-1) == self.d:
                out_learn = torch.bmm(self._D_inv_sqrt, out_learn)
            else:
                out_learn = self._D_inv_sqrt * out_learn

        # Fixed channels
        out_fixed = self._apply_fixed_channels(H_fixed, row, col)

        out = out_learn if out_fixed is None else torch.cat([out_learn, out_fixed], dim=1)
        return self._restore_output(out, meta)


class LazyBundleLaplacianBuilder(LazyLaplacianBuilder):
    """
    Lazy Sheaf Laplacian with orthogonal (bundle) restriction maps.

    For bundle sheaves, F_{v▷e} ∈ O(d).  Since F^T F = I the block-diagonal
    degree matrix is simply deg(v) · I, making normalization trivial.
    """

    def __init__(
        self,
        size,
        edge_index,
        d,
        add_hp=False,
        add_lp=False,
        orth_map="cayley",
        augmented=True,
    ):
        # Bundle sheaves always use symmetric (block) normalization.
        super(LazyBundleLaplacianBuilder, self).__init__(
            size,
            edge_index,
            d,
            normalised=True,
            deg_normalised=False,
            add_hp=add_hp,
            add_lp=add_lp,
            augmented=augmented,
        )
        self.orth_transform = Orthogonal(d=self.d, orthogonal_map=orth_map)
        self.orth_map = orth_map
        self._maps = None

    # Override so that the correct constructor arguments are forwarded.
    def create_with_new_edge_index(self, edge_index):
        assert edge_index.max() <= self.size
        new_builder = LazyBundleLaplacianBuilder(
            self.size,
            edge_index,
            self.d,
            add_hp=self.add_hp,
            add_lp=self.add_lp,
            orth_map=self.orth_map,
            augmented=self.augmented,
        )
        new_builder.train(self.training)
        return new_builder

    def forward(self, map_params, edge_weights=None):
        """
        Process orthogonal map parameters.

        Args:
            map_params: Parameters [m, k] → orthogonal, or pre-computed [m, d, d]
            edge_weights: Optional [m, 1]

        Returns:
            (orthogonal_maps, saved_maps)
        """
        if map_params.dim() == 2:
            maps = self.orth_transform(map_params)  # [m, d, d]
        else:
            assert map_params.dim() == 3
            maps = map_params

        if edge_weights is not None:
            maps = maps * edge_weights.unsqueeze(-1)

        self._maps = maps

        # For O(d) maps, D_v = deg(v) · I  →  D^{-1/2} = (deg+α)^{-1/2} · I
        if self.augmented:
            self._D_inv_sqrt = (self.deg + 1).pow(-0.5).view(-1, 1, 1)
        else:
            d_inv = self.deg.pow(-0.5)
            d_inv.masked_fill_(d_inv == float("inf"), 0)
            self._D_inv_sqrt = d_inv.view(-1, 1, 1)

        saved_maps = maps.detach().clone()
        return maps, saved_maps

    def apply(self, H):
        """Compute normalised L_F @ H for bundle sheaves."""
        if self._maps is None:
            raise RuntimeError("Must call forward() with maps before apply()")

        H, meta = self._reshape_input(H)

        row, col = self.edge_index
        O = self._maps  # [m, d, d]

        H_learn = H[:, : self.d, :]
        H_fixed = H[:, self.d :, :]

        # Pre-scale (scalar normalization for bundle)
        H_scaled = self._D_inv_sqrt * H_learn  # [n, d, c]

        h_src = H_scaled[row]
        h_dst = H_scaled[col]

        # Coboundary + adjoint
        Oh_src = torch.bmm(O, h_src)
        Oh_dst = torch.bmm(O, h_dst)
        delta = Oh_src - Oh_dst
        Ot_delta = torch.bmm(O.transpose(-1, -2), delta)

        out_learn = torch.zeros_like(H_scaled)
        out_learn.scatter_add_(0, row.view(-1, 1, 1).expand_as(Ot_delta), Ot_delta)

        # Post-scale
        out_learn = self._D_inv_sqrt * out_learn

        # Fixed channels
        out_fixed = self._apply_fixed_channels(H_fixed, row, col)

        out = out_learn if out_fixed is None else torch.cat([out_learn, out_fixed], dim=1)
        return self._restore_output(out, meta)


# =============================================================================
# Wrapper for drop-in replacement
# =============================================================================


class LazyLaplacianOp:
    """
    Callable object representing the lazy Laplacian operator.

    Supports ``L(x)`` and ``L @ x``.
    """

    def __init__(self, builder):
        self.builder = builder

    def __call__(self, x):
        return self.builder.apply(x)

    def __matmul__(self, x):
        return self.builder.apply(x)


class LazyLaplacianWrapper(nn.Module):
    """
    Wrapper that provides the same interface as an explicit LaplacianBuilder
    but uses lazy evaluation internally.

        # Before:
        L, maps = self.laplacian_builder(maps)
        out = torch_sparse.spmm(L[0], L[1], N, N, x)

        # After (with wrapper):
        L, maps = self.laplacian_builder(maps)   # L is a LazyLaplacianOp
        out = L(x)
    """

    def __init__(self, lazy_builder):
        super().__init__()
        self.lazy_builder = lazy_builder

    def forward(self, maps, *args, **kwargs):
        processed_maps, saved_maps = self.lazy_builder(maps, *args, **kwargs)
        lazy_op = LazyLaplacianOp(self.lazy_builder)
        return lazy_op, saved_maps

    def apply_laplacian(self, H):
        """Explicit name to avoid shadowing nn.Module.apply."""
        return self.lazy_builder.apply(H)

    def update_graph(self, edge_index, size=None):
        return self.lazy_builder.update_graph(edge_index, size)


# =============================================================================
# Factory function
# =============================================================================


def create_lazy_builder(builder_type, size, edge_index, d, **kwargs):
    """
    Factory function to create lazy builders.

    Args:
        builder_type: One of 'diag', 'general', 'bundle'
        size: Number of nodes
        edge_index: Edge index tensor
        d: Stalk dimension
        **kwargs: Additional arguments passed to builder

    Returns:
        LazyLaplacianBuilder instance
    """
    _BUILDERS = {
        "diag": LazyDiagLaplacianBuilder,
        "diagonal": LazyDiagLaplacianBuilder,
        "general": LazyGeneralLaplacianBuilder,
        "bundle": LazyBundleLaplacianBuilder,
        "orthogonal": LazyBundleLaplacianBuilder,
    }

    key = builder_type.lower()
    if key not in _BUILDERS:
        raise ValueError(f"Unknown builder type: {builder_type}. Choose from {list(_BUILDERS.keys())}")

    cls = _BUILDERS[key]

    # Filter kwargs to only those accepted by the target class's __init__
    # to prevent silent mismatches (e.g. passing normalised= to Bundle).
    import inspect

    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}

    ignored = set(kwargs.keys()) - set(filtered.keys())
    if ignored:
        import warnings
        warnings.warn(
            f"create_lazy_builder: ignoring kwargs not accepted by "
            f"{cls.__name__}: {ignored}"
        )

    return cls(size, edge_index, d, **filtered)