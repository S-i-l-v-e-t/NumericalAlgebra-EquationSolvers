from typing import Any
from torch import Tensor
import torch
from __genericSolver import Solver

class JacobiSolver(Solver):
    def __init__(self, device, max_iter : int = 1000, tol : float = 1e-6, omega : float = 1.00, *args: Any, **kwds: Any) -> None:
        super().__init__(device, *args, **kwds)
        self.tol = tol
        self.max_iter = max_iter
        self.omega = omega
    
    def __str__(self) -> str:
        return f"JacobiSolver: tol={self.tol}, max_iter={self.max_iter}, omega={self.omega}, device={self.device}"
    
    def _process_data(self, mat: Tensor) -> Tensor:
        mat = super()._process_data(mat)
        matA = mat[:, :-1]
        diag = torch.diag(matA)
        assert not (diag == 0).any(), "Jacobi Solver Requires Non-zero Diagonal Elements"
        row_sums = torch.sum(torch.abs(matA), dim=1) - torch.abs(diag)
        if not torch.all(torch.abs(diag) > row_sums):
            print("User Warning: Matrix is not strictly diagonally dominant. Jacobi iteration may NOT converge!")
        return mat
    
    def _solve(self, mat: Tensor) -> Tensor:
        matA = mat[:, :-1]
        matb = mat[:, -1]
        n = matA.shape[0]
        diag = torch.diagonal(matA)
        x = torch.zeros(n, dtype=mat.dtype, device=mat.device)
        for _ in range(self.max_iter):
            r = matb - torch.matmul(matA, x)
            if torch.max(torch.abs(r)) < self.tol:
                return x
            x = x + ((r / diag)*self.omega)
        print(f"User Warning: Jacobi solver reached max_iter ({self.max_iter}) without full convergence.")
        return x