from __future__ import annotations

import uvicorn

from .app import create_app
from .config import get_settings

app = create_app()


def main() -> None:
    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port, reload=False)


if __name__ == '__main__':
    main()

