from typing import Any
from __genericSolver import Solver
from torch import Tensor
import torch

class CholeskySolver(Solver):
    def __init__(self, device, *args: Any, **kwds: Any) -> None:
        super().__init__(device, *args, **kwds)
    def _solve(self, mat: Tensor) -> Tensor:
        matA = mat[:,:-1]
        matb = mat[:,-1]
        n = matA.shape[0]
        try:
            L = torch.linalg.cholesky(matA)
        except RuntimeError:
            raise AssertionError("Cholesky Solver Requires a Symmetric Positive Definite Matrix")
        x = matb.clone()
        for i in range(n):
            x[i] = (x[i] - torch.dot(L[i, :i], x[:i])) / L[i, i]
        L_T = L.T
        for i in range(n - 1, -1, -1):
            x[i] = (x[i] - torch.dot(L_T[i, i+1:], x[i+1:])) / L_T[i, i]
            
        return x