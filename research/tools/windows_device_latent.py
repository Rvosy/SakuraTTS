"""Experimental GPU-resident split latent, using ORT and the CUDA runtime only.

No kernel compilation or CuPy is needed. CUDA 2D copies pack channel-major
latent slices into owned contiguous CUDA buffers; chunk waveform reconstruction stays
on the CPU exactly as in the host-transfer control.
"""
import ctypes
import sys
import time

import numpy as np

from windows_chunked_synthesis import SplitAcousticAdapter


class DevicePointer:
    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer


def copy_layout(shape, start, end, itemsize):
    if (len(shape) != 3 or shape[0] != 1 or min(shape) < 1
            or not 0 <= start < end <= shape[2] or itemsize not in (2, 4)):
        raise ValueError("Require a nonempty channel-major latent slice")
    return {"source_offset_bytes": start * itemsize, "source_pitch_bytes": shape[2] * itemsize,
            "destination_pitch_bytes": (end - start) * itemsize,
            "width_bytes": (end - start) * itemsize, "height": shape[1]}


class CudaCopies:
    def __init__(self):
        self.library = ctypes.WinDLL("cudart64_12.dll")
        self.library.cudaMemcpy2D.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int]
        self.library.cudaMemcpy2D.restype = ctypes.c_int
        self.library.cudaDeviceSynchronize.argtypes = []
        self.library.cudaDeviceSynchronize.restype = ctypes.c_int
        self.library.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.library.cudaGetErrorString.restype = ctypes.c_char_p
        self.library.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.library.cudaGetDevice.restype = ctypes.c_int
        self.library.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.library.cudaMalloc.restype = ctypes.c_int
        self.library.cudaFree.argtypes = [ctypes.c_void_p]
        self.library.cudaFree.restype = ctypes.c_int
        self.library.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.library.cudaMemcpy.restype = ctypes.c_int
        device = ctypes.c_int()
        self.check(self.library.cudaGetDevice(ctypes.byref(device)))
        if device.value != 0:
            raise RuntimeError("This experiment requires the current CUDA device to be zero")

    def check(self, status):
        if status:
            raise RuntimeError(self.library.cudaGetErrorString(status).decode("utf-8", errors="replace"))

    def synchronize(self):
        self.check(self.library.cudaDeviceSynchronize())

    def allocate(self, size):
        pointer = ctypes.c_void_p()
        self.check(self.library.cudaMalloc(ctypes.byref(pointer), size))
        return pointer.value

    def upload(self, pointer, array):
        self.check(self.library.cudaMemcpy(pointer, array.ctypes.data, array.nbytes, 1))
        self.synchronize()

    def release(self, pointer):
        self.check(self.library.cudaFree(pointer))

    def pack(self, destination, source, layout):
        self.check(self.library.cudaMemcpy2D(destination.data_ptr(), layout["destination_pitch_bytes"],
            source.data_ptr() + layout["source_offset_bytes"], layout["source_pitch_bytes"],
            layout["width_bytes"], layout["height"], 3))  # cudaMemcpyDeviceToDevice
        # ORT's two sessions and this CUDA-runtime call may use different streams.
        # Establish completion explicitly before the consumer session reads it.
        self.synchronize()


class DeviceLatentAdapter(SplitAcousticAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.copies = CudaCopies()
        self.runtime.update(latent_transfer="cuda-iobinding-cudart-copy2d",
                            synchronization="explicit CUDA completion before cross-session consumption")

    def decode(self, codes, phones, ge, ge512, noise, *, noise_scale=.5, speed=1., capture=False):
        if self.session is None or self.vocoder_session is None:
            raise RuntimeError("The split acoustic model has been unloaded")
        if capture:
            raise ValueError("Intermediate capture is unsupported by the device-latent experiment")
        self.last_transfer = None
        feeds = self._inputs(codes, phones, ge, ge512, noise, noise_scale, speed)
        latent_binding = vocoder_binding = latent = chunk = None
        scratch_pointer = condition_pointer = None
        try:
            started = time.perf_counter()
            latent_binding = self.session.io_binding()
            for name, value in feeds.items():
                latent_binding.bind_cpu_input(name, value)
            latent_binding.bind_output("decoder_input", "cuda", 0)
            self.session.run_with_iobinding(latent_binding, self._run_options)
            latent_binding.synchronize_outputs()
            latent = latent_binding.get_outputs()[0]
            self.copies.synchronize()
            latent_ms = (time.perf_counter() - started) * 1000
            config = self.encoder.manifest["config"]
            total = feeds["codes"].shape[-1] * config["semantic_upsample_factor"]
            dtype = np.float16 if self.encoder.manifest["dtype"] == "float16" else np.float32
            if (latent.device_name() != "cuda" or latent.shape() != [1, config["model"]["inter_channels"], total]
                    or latent.data_type() != ("tensor(float16)" if dtype == np.float16 else "tensor(float)")):
                raise RuntimeError("Expected the complete internal latent on CUDA")
            ratio = self.planner.samples_per_frame
            waveform = np.empty((1, 1, total * ratio), np.float32)
            plans = ([self.planner.plan(total, 0, total)] if self.chunk_frames == 0 else
                     self.planner.plan_chunks(total, self.chunk_frames))
            condition_pointer = self.copies.allocate(feeds["ge"].nbytes)
            self.copies.upload(condition_pointer, feeds["ge"])
            scratch_bytes = max((plan["input_end"]-plan["input_start"]) for plan in plans) * latent.shape()[1] * np.dtype(dtype).itemsize
            if len(plans) > 1:
                scratch_pointer = self.copies.allocate(scratch_bytes)
            scratch = DevicePointer(scratch_pointer)
            copied_bytes = outputs_bytes = 0
            for plan in plans:
                start, end = plan["input_start"], plan["input_end"]
                layout = copy_layout(latent.shape(), start, end, np.dtype(dtype).itemsize)
                if start == 0 and end == total:
                    packed_pointer = latent.data_ptr()
                else:
                    self.copies.pack(scratch, latent, layout)
                    packed_pointer = scratch_pointer
                    copied_bytes += layout["width_bytes"] * layout["height"]
                vocoder_binding = self.vocoder_session.io_binding()
                vocoder_binding.bind_input("decoder_input", "cuda", 0, dtype,
                                           [1, latent.shape()[1], end-start], packed_pointer)
                vocoder_binding.bind_input("ge", "cuda", 0, np.float32, list(feeds["ge"].shape), condition_pointer)
                vocoder_binding.bind_output("waveform", "cpu")
                self.vocoder_session.run_with_iobinding(vocoder_binding, self._run_options)
                vocoder_binding.synchronize_outputs()
                chunk = vocoder_binding.get_outputs()[0].numpy()
                if chunk.dtype != np.float32 or chunk.shape != (1, 1, (end-start)*ratio) or not np.isfinite(chunk).all():
                    raise RuntimeError("Device-latent vocoder produced an invalid chunk")
                a, b = plan["core_start_frame"], plan["core_end_frame"]
                waveform[..., a*ratio:b*ratio] = chunk[..., plan["crop_start"]:plan["crop_end"]]
                outputs_bytes += chunk.nbytes
                vocoder_binding.clear_binding_inputs()
                vocoder_binding.clear_binding_outputs()
                vocoder_binding = chunk = None
            self.last_transfer = {"experiment": "device-latent-vocoder", "chunk_frames": self.chunk_frames,
                "chunks": len(plans), "latent_dtype": str(np.dtype(dtype)), "latent_frames": total,
                "latent_d2h_bytes": 0, "latent_inputs_h2d_bytes": sum(value.nbytes for value in feeds.values()),
                "vocoder_inputs_h2d_bytes": feeds["ge"].nbytes, "latent_pack_d2d_bytes": copied_bytes,
                "scratch_allocated_bytes": scratch_bytes if scratch_pointer is not None else 0,
                "allocation": "Request-scoped cudaMalloc scratch and condition; no OrtValue factory global arena",
                "vocoder_outputs_d2h_bytes": outputs_bytes, "latent_ms": latent_ms,
                "decode_ms": (time.perf_counter()-started)*1000, "plans": plans,
                "byte_scope": "Logical explicit transfers; excludes runtime-internal copies and workspaces"}
            return waveform
        finally:
            original_error = sys.exc_info()[1]
            cleanup_errors = []
            for binding in (vocoder_binding, latent_binding):
                if binding is not None:
                    try:
                        binding.clear_binding_inputs()
                        binding.clear_binding_outputs()
                    except Exception as error:
                        cleanup_errors.append(error)
            feeds.clear()
            latent_binding = vocoder_binding = latent = chunk = None
            for pointer in (scratch_pointer, condition_pointer):
                if pointer is not None:
                    try:
                        self.copies.release(pointer)
                    except Exception as error:
                        cleanup_errors.append(error)
            if cleanup_errors:
                self.last_transfer = None
                if original_error is not None:
                    if hasattr(original_error, "add_note"):
                        original_error.add_note(f"CUDA scratch cleanup failed: {cleanup_errors!r}")
                else:
                    raise RuntimeError(f"CUDA scratch cleanup failed: {cleanup_errors!r}")
            if self.last_transfer is not None:
                self.last_transfer["decode_ms"] = (time.perf_counter()-started)*1000
