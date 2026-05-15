from typing import Any
from torch import Tensor
import torch
from Solvers.__genericSolver import Solver

class GaussSeidelSolver(Solver):
    def __init__(self, device=None, max_iter : int = 1000, tol : float = 1e-6, omega : float = 1.00, *args: Any, **kwds: Any) -> None:
        super().__init__(device, *args, **kwds)
        self.tol = tol
        self.max_iter = max_iter
        self.omega = omega
    
    def __str__(self) -> str:
        return f"GaussSeidelSolver: tol={self.tol}, max_iter={self.max_iter}, omega={self.omega}, device={self.device}"
    
    def _process_data(self, mat: Tensor) -> Tensor:
        mat = super()._process_data(mat)
        matA = mat[:, :-1]
        diag = torch.diag(matA)
        assert not (diag == 0).any(), "Gauss-Seidel Solver Requires Non-zero Diagonal Elements"
        row_sums = torch.sum(torch.abs(matA), dim=1) - torch.abs(diag)
        if not torch.all(torch.abs(diag) > row_sums):
            print("User Warning: Matrix is not strictly diagonally dominant. Gauss-Seidel iteration may NOT converge!")
        return mat
    
    def _solve_traditional(self, matA: Tensor, matb: Tensor, n: int) -> Tensor:
        if self.device != "cpu":
            matA = matA.cpu()
            matb = matb.cpu()
        x = torch.zeros(n, dtype=torch.float64, device='cpu')
        for _ in range(self.max_iter):
            x_old = x.clone()
            for i in range(n):
                s1 = torch.dot(matA[i, :i], x[:i])
                s2 = torch.dot(matA[i, i+1:], x[i+1:])
                x[i] = (1 - self.omega) * x[i] + self.omega * ((matb[i] - s1 - s2) / matA[i, i])
                
            if torch.max(torch.abs(x - x_old)) < self.tol:
                break

        return x.to(self.device)
    
    def _solve_splitting(self, matA: Tensor, matb: Tensor, n: int) -> Tensor:
        L = torch.tril(matA, diagonal=-1)
        U = torch.triu(matA, diagonal=1)
        D = torch.diag(torch.diag(matA))
        x = torch.zeros(n, dtype=matA.dtype, device=matA.device)
        b_col = matb.unsqueeze(1) 
        for _ in range(self.max_iter):
            x_old = x.clone()
            rhs = self.omega * b_col + torch.matmul((1 - self.omega) * D - self.omega * U, x.unsqueeze(1))
            x_new_col = torch.linalg.solve_triangular((L * self.omega) + D, rhs, upper=False)
            x = x_new_col.squeeze(1) 
            if torch.max(torch.abs(x - x_old)) < self.tol:
                break
        return x
    
    def _solve(self, mat: Tensor) -> Tensor:
        matA = mat[:,:-1]
        matb = mat[:,-1]
        n = mat.shape[0]
        if self.device == "cpu":
            return self._solve_traditional(matA,matb,n)
        return self._solve_splitting(matA,matb,n)