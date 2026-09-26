"""ONNX autoregressive GPT on a DirectML adapter.

The Transformer runs in DirectML. Tokenization, embeddings, sampling and the
bounded host KV buffers use the CPU; FP16 graphs retain FP16 KV storage.
"""

from sakuratts.backends.cpu.onnx_gpt import ONNXCPUGPT


class DirectMLGPT(ONNXCPUGPT):
    """Use the shared graph contract with a separate GPU execution session."""

    device = "directml"

    def _create_session(self, graph, options):
        import onnxruntime as ort

        if "DmlExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("DirectML GPT requires onnxruntime-directml")
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(graph), sess_options=options, enable_fallback=False,
            providers=[("DmlExecutionProvider", {"device_id": str(self.device_id)}),
                       "CPUExecutionProvider"])
        try:
            if session.get_providers()[0] != "DmlExecutionProvider":
                raise RuntimeError("DirectML GPT did not activate DmlExecutionProvider")
            return session
        except BaseException:
            session = None
            raise
