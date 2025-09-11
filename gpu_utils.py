"""Utility for leveraging all available CUDA GPUs in PyTorch models.

This module exposes :class:`CudaMultiGPU`, a lightweight wrapper that detects
all CUDA devices on the host and moves a model to run on them in parallel via
``torch.nn.DataParallel``.  It can be imported and used in any project without
modification.

Example
-------
Wrap a model so that it automatically utilizes every available GPU:

>>> from gpu_utils import CudaMultiGPU
>>> model = MyModel()
>>> mgpu = CudaMultiGPU(model)
>>> output = mgpu(input_tensor)  # ``mgpu`` is callable

Alternatively access the wrapped model explicitly:

>>> wrapped_model = mgpu.model
>>> output = wrapped_model(input_tensor)
"""

from __future__ import annotations

import torch


class CudaMultiGPU:
    """Prepare a model to use all available CUDA GPUs.

    This class detects all CUDA devices on the host machine and, when CUDA is
    available, wraps the provided :class:`torch.nn.Module` in
    :class:`torch.nn.DataParallel` so that the model runs in parallel on all
    GPUs.  If CUDA is not available, the model is left on the CPU.

    Notes
    -----
    The :class:`CudaMultiGPU` instance is itself callable and forwards all
    provided arguments to the wrapped model.  The underlying model can also be
    accessed through the :attr:`model` attribute if direct access is preferred.
    """

    def __init__(self, model: torch.nn.Module, verbose: bool = True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if torch.cuda.is_available():
            if self.device_count > 1:
                model = torch.nn.DataParallel(model, device_ids=list(range(self.device_count)))
            model = model.to(self.device)
            if verbose:
                print(f"Using {self.device_count} CUDA device(s).")
        elif verbose:
            print("CUDA not available. Running on CPU.")

        self.model = model

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


if __name__ == "__main__":  # pragma: no cover - usage demonstration
    class _ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    example_model = _ToyModel()
    mgpu = CudaMultiGPU(example_model)
    sample_input = torch.randn(1, 4).to(mgpu.device)
    print(mgpu(sample_input))
