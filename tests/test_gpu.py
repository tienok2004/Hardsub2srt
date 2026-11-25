"""GPU startup must load pip-installed CUDA libraries before opening a session."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core import gpu


class GpuTests(unittest.TestCase):
    def test_preloads_cuda_libraries_before_creating_session(self) -> None:
        calls = []

        def preload_dlls() -> None:
            calls.append("preload")

        def inference_session(*args, **kwargs):
            calls.append("session")
            return SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"])

        ort = SimpleNamespace(
            get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
            preload_dlls=preload_dlls,
            InferenceSession=inference_session,
        )
        with patch.object(gpu, "_onnxruntime", return_value=ort), patch.object(gpu, "_device_name", return_value="GPU"):
            status = gpu.inspect_gpu("model.onnx")

        self.assertTrue(status.cuda_session_initialized)
        self.assertEqual(calls, ["preload", "session"])


if __name__ == "__main__":
    unittest.main()
