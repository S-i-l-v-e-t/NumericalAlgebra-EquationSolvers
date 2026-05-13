
<div align="center">
  <h1>NumericalAlgebra-EquationSolvers</h1>
  <p>自建线性方程组求解器集合，使用Pytorch提供GPU并行化支持。</p>
  <p>Homebaked linear equation system solvers, using pytorch for supporting GPU computation.</p>
</div>

## 数值方法使用实例/Numerical Method Instance
```Python
import JacobiSolver from Solvers.Jacobi 
import torch
A = torch.tensor([
    ...
])

j_solver = JacobiSolver(device="cuda", max_iter = 1000, tol = 1e-6, omega = 1.00)
j_solver(matrix=A)
```
- device : [torch.device](https://docs.pytorch.org/docs/2.11/tensor_attributes.html#torch-device) 实际进行计算的设备,缺省优先CUDA
- max_iter : int 最大循环上限,缺省为1000
- tol : float 可容忍误差,缺省为1e-6
- omega : float 松弛因子,缺省为1.00(非松弛)
- *matrix [torch.Tensor](https://docs.pytorch.org/docs/2.11/tensors.html#torch-tensor) 待求解方程组的增广矩阵A

## 直接方法使用实例/Direct Method Instance
```Python
import GaussSolver from Solvers.Gauss
import torch
A = torch.tensor([
    ...
])

g_solver = GaussSolver(device = "cuda")
g_solver(matrix = A)
```
- device : [torch.device](https://docs.pytorch.org/docs/2.11/tensor_attributes.html#torch-device) 实际进行计算的设备,缺省优先CUDA
- *matrix [torch.Tensor](https://docs.pytorch.org/docs/2.11/tensors.html#torch-tensor) 待求解方程组的增广矩阵A