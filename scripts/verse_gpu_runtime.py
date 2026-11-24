"""Select this project's GPU environment when launching a CUDA CLI directly."""
import os
from pathlib import Path
import sys


def configure_cuda_cli():
    # The shared runner can select the matching user-space driver only before
    # the dynamic loader starts Python. Keep the caller's interpreter and flags.
    if os.environ.get("ILM_GPU_LAUNCHED") == "1":
        return
    launcher = Path(__file__).with_name("with_ilm_gpu")
    env = {**os.environ, "ILM_GPU_LAUNCHED":"1"}
    os.execve(launcher, [str(launcher), sys.executable, *sys.orig_argv[1:]], env)
