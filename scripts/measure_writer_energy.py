"""Read NVIDIA's physical device energy counter for a GPU owned by this experiment."""

import argparse
import ctypes
import json
import os
import subprocess
import sys


def read_counter(index):
    lib = ctypes.CDLL("libnvidia-ml.so.1")

    def check(code):
        if code:
            raise OSError(f"NVML status {code}")

    check(lib.nvmlInit_v2())
    try:
        handle = ctypes.c_void_p()
        value = ctypes.c_ulonglong()
        check(lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)))
        check(lib.nvmlDeviceGetTotalEnergyConsumption(handle, ctypes.byref(value)))
        uuid = ctypes.create_string_buffer(96)
        check(lib.nvmlDeviceGetUUID(handle, uuid, len(uuid)))
        return dict(millijoules=value.value, uuid=uuid.value.decode())
    finally:
        lib.nvmlShutdown()


def snapshot():
    # An optional driver-environment wrapper is useful on a shared host during
    # a rolling upgrade. Keep its library environment out of the training process.
    prefix = os.environ.get("ILM_NVML_RUNNER")
    command = ([prefix] if prefix else []) + [
        sys.executable,
        "-m",
        "scripts.measure_writer_energy",
    ]
    r = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=12
    )
    if r.returncode:
        return dict(
            error=r.stderr.strip().splitlines()[-1]
            if r.stderr.strip()
            else "unavailable"
        )
    return json.loads(r.stdout)


def consumed(a, b):
    if "error" in a or "error" in b:
        return dict(joules=None, reason=a.get("error") or b.get("error"))
    if a["uuid"] != b["uuid"] or b["millijoules"] < a["millijoules"]:
        raise ValueError("Device identity or counter changed")
    return dict(joules=(b["millijoules"] - a["millijoules"]) / 1000, device=a["uuid"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0)
    print(json.dumps(read_counter(ap.parse_args().gpu)))
