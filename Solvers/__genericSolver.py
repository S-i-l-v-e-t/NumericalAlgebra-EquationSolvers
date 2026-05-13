from typing import Any
import torch


class Solver():
    def __init__(self, device, *args: Any, **kwds: Any) -> None:
        """ Initialize Solver """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
    def __call__(self, matrix: torch.Tensor, *args: Any, **kwds: Any) -> torch.Tensor:
        """
        Call Auto Solve to Solve matrix as a Augmented matrix
        matrix -> tensor
        """
        return self._solve(self._process_data(matrix).to(self.device))
    def __str__(self) -> str:
        """
        Useful debug infomation of solver
        """
        return ""
    def _process_data(self,mat: torch.Tensor) -> torch.Tensor:
        """
        basically process data which match the solve method's requires
        """
        assert len(mat.shape) == 2 and mat.shape[1] == mat.shape[0] + 1, "Invaild Argumented Matrix Shape"
        return mat.to(torch.float64)
    def _solve(self,mat: torch.Tensor) -> torch.Tensor:
        """
        major processing steps to solve the equation
        """
        return mat

