from typing import Optional

from .core import HugDown


def get(repo: str,
        output_dir: str,
        data_files: Optional[str] = None,
        num_proc: Optional[int] = None,
        token: Optional[str] = None,
        log_level: Optional[int] = None):
    huggingface_dl = HugDown(num_proc=num_proc, token=token, log_level=log_level)
    huggingface_dl.preload_files(repo=repo, data_files=data_files)
    huggingface_dl.start(output_dir=output_dir)
