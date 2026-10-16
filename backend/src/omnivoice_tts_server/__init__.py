import os

# Before anything imports torch: number CUDA devices like nvidia-smi (PCI bus order), so
# cuda:N in preferred_device is nvidia-smi GPU N. CUDA's default (fastest first) can
# swap the cards on a mixed two-GPU box. An explicit environment value still wins.
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

from .app import create_app  # noqa: E402

__all__ = ["create_app"]
