from typing import Any
from torch import Tensor
import torch
from Solvers.__genericSolver import Solver

class GaussSolver(Solver):
    def __init__(self, device=None, *args: Any, **kwds: Any) -> None:
        super().__init__(device, *args, **kwds)
    def _solve(self, mat: Tensor) -> Tensor:
        n = mat.shape[0]
        for k in range(n):
            max_idx = torch.argmax(torch.abs(mat[k:, k])) + k
            if max_idx != k:
                mat[[k, max_idx]] = mat[[max_idx, k]]
            factors = mat[k+1:, k] / mat[k, k]
            mat[k+1:, k:] -= factors.unsqueeze(1) * mat[k, k:]
        
        if not torch.isfinite(mat).all().item():
            print("User Waring: Singular Matrix Detected, Result May Contains INF/NAN")
        
        x = torch.empty(n, dtype=mat.dtype, device=mat.device)
        for i in range(n - 1, -1, -1):
            x[i] = (mat[i, -1] - torch.dot(mat[i, i+1:n], x[i+1:n])) / mat[i, i]
        return x