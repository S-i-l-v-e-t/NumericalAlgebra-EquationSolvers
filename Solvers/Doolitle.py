from typing import Any
import torch
from torch import Tensor
from Solvers.__genericSolver import Solver

class DoolitleSolver(Solver):
    def __init__(self, device=None, *args: Any, **kwds: Any) -> None:
        super().__init__(device, *args, **kwds)
    def _solve(self, mat: Tensor) -> Tensor:
        matA = mat[:,:-1]
        matb = mat[:,-1]
        matA,matb = mat
        n = matA.shape[0]
        P,L,U = torch.linalg.lu(matA)
        matb = P @ matb
        y = torch.empty(n, dtype=torch.float64, device=self.device)
        for i in range(n):
            y[i] = matb[i] - torch.dot(L[i, :i], y[:i])
            
        x = torch.empty(n, dtype=torch.float64, device=self.device)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - torch.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        return x