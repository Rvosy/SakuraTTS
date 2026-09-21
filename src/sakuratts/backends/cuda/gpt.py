"""Single-weight CUDA GPT executor with in-place KV and optional CUDA graph.

CuPy/cuBLAS execute the existing GPT package. Sampling and stopping remain in
generation.py. No Torch, training source or alternate backend is imported.
"""

import ctypes
import json
import os
from pathlib import Path

import numpy as np

from sakuratts.backends.cuda.runtime import import_cupy, validate_gpt_cuda_include_paths
from sakuratts._internal.reference_condition import sha256_file
from sakuratts._internal.weight_storage import read_fp32, validate_storage

validate_gpt_cuda_include_paths()
cp = import_cupy()


class _GraphBLAS:
    """cuBLAS handle bound once to our stream; no CuPy capture-time setStream.

    CuPy 14 deliberately rejects its BLAS wrappers during stream capture.
    The public CUDA cuBLAS API supports it when the handle is configured before
    capture. Matrices remain owned by CuPy; this adapter never allocates weights.
    """
    def __init__(self, stream, precision="fp32"):
        self.lib = ctypes.CDLL("cublas64_12.dll" if os.name == "nt" else "libcublas.so.12")
        self.handle = ctypes.c_void_p()
        self.lib.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
        self.lib.cublasSgemm_v2.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.lib.cublasGemmEx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.cublasGemmStridedBatchedEx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_int, ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self._check(self.lib.cublasCreate_v2(ctypes.byref(self.handle)))
        self._check(self.lib.cublasSetStream_v2(self.handle, ctypes.c_void_p(stream.ptr)))
        if precision == "fp16":
            self.lib.cublasSetMathMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
            # Prevent a split-K reduction from rounding partial sums to FP16.
            self._check(self.lib.cublasSetMathMode(self.handle, 16))
        self.alpha, self.beta = ctypes.c_float(1), ctypes.c_float(0)

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"cuBLAS failed with status {status}")

    def linear(self, x, weight, out):
        rows, columns = weight.shape
        if x.dtype == cp.float16:
            self._gemm_fp16(x, weight, out, transpose_y=True)
            return
        self._check(self.lib.cublasSgemm_v2(self.handle, 1, 0, rows, 1, columns,
            ctypes.byref(self.alpha), weight.data.ptr, columns, x.data.ptr, columns,
            ctypes.byref(self.beta), out.data.ptr, rows))

    def _gemm_fp16(self, x, y, out, *, transpose_y=False):
        """Row-major FP16 GEMM, FP32 accumulation and FP16 or FP32 output."""
        if (x.dtype != cp.float16 or y.dtype != cp.float16
                or out.dtype not in (cp.float16, cp.float32)):
            raise ValueError("FP16 GEMM requires FP16 operands and FP16/FP32 output")
        if not all(a.flags.c_contiguous for a in (x, y, out)):
            raise ValueError("FP16 GEMM requires contiguous row-major operands")
        if x.ndim not in (2, 3) or y.ndim != x.ndim or out.ndim != x.ndim:
            raise ValueError("FP16 GEMM requires matching 2D or 3D operands")
        m, k = x.shape[-2:]
        n = y.shape[-2] if transpose_y else y.shape[-1]
        if (y.shape[-1 if transpose_y else -2] != k
                or out.shape != (*x.shape[:-2], m, n)
                or y.shape[:-2] != x.shape[:-2]):
            raise ValueError("FP16 GEMM operand shapes do not align")
        # CUDA_R_16F=2, CUDA_R_32F=0, CUBLAS_COMPUTE_32F=68.
        # Transposing the product lets column-major cuBLAS read row-major data.
        args = (self.handle, int(transpose_y), 0, n, m, k,
                ctypes.byref(self.alpha), y.data.ptr, 2, y.shape[-1])
        output_type = 2 if out.dtype == cp.float16 else 0
        if x.ndim == 2:
            self._check(self.lib.cublasGemmEx(*args, x.data.ptr, 2, k,
                ctypes.byref(self.beta), out.data.ptr, output_type, n, 68, -1))
        else:
            self._check(self.lib.cublasGemmStridedBatchedEx(*args,
                y.shape[-2] * y.shape[-1], x.data.ptr, 2, k, m*k,
                ctypes.byref(self.beta), out.data.ptr, output_type, n, m*n,
                x.shape[0], 68, -1))

    def close(self):
        if self.handle:
            self._check(self.lib.cublasDestroy_v2(self.handle))
            self.handle = ctypes.c_void_p()


_SOURCE = r'''
extern "C" __global__ void layer_norm(float* x, const float* residual,
 const float* bias, const float* weight, const float* shift, int width, float eps) {
  __shared__ float sums[256];
  int row=blockIdx.x, tid=threadIdx.x;
  float s=0;
  for(int c=tid;c<width;c+=256) s+=x[row*width+c]+residual[row*width+c]+bias[c];
  sums[tid]=s; __syncthreads();
  for(int n=128;n;n>>=1){if(tid<n)sums[tid]+=sums[tid+n];__syncthreads();}
  float mean=sums[0]/width; s=0;
  for(int c=tid;c<width;c+=256){float v=x[row*width+c]+residual[row*width+c]+bias[c]-mean;s+=v*v;}
  sums[tid]=s; __syncthreads();
  for(int n=128;n;n>>=1){if(tid<n)sums[tid]+=sums[tid+n];__syncthreads();}
  float inv=rsqrtf(sums[0]/width+eps);
  for(int c=tid;c<width;c+=256) x[row*width+c]=((x[row*width+c]+residual[row*width+c]+bias[c])-mean)*inv*weight[c]+shift[c];
}
extern "C" __global__ void embedding(float* x,const float* emb,const float* pe,
 const float* alpha,const int* state,int width) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;
 if(i<width)x[i]=emb[state[0]*width+i]+alpha[0]*pe[state[2]*width+i];
}
extern "C" __global__ void kv_write(const float* qkv, const float* bias,
 float* key,float* value,const int* state,int width,int dim,int capacity){
 int i=blockDim.x*blockIdx.x+threadIdx.x;
 if(i<width){int dest=(i/dim*capacity+state[1])*dim+i%dim;
 key[dest]=qkv[width+i]+bias[width+i];value[dest]=qkv[2*width+i]+bias[2*width+i];}
}
extern "C" __global__ void attention(const float* qkv,const float* bias,
 const float* key,const float* value,float* out,const int* state,int dim,int capacity){
 extern __shared__ float scores[];
 __shared__ float reduce[256];
 int h=blockIdx.x,tid=threadIdx.x,n=state[1]+1;
 int lane=tid%32,warp=tid/32;
 // One warp reads one key row contiguously. The previous serial-dot layout
 // made adjacent lanes fetch different rows and wasted memory transactions.
 for(int p=warp;p<n;p+=8){float s=0;
   for(int d=lane;d<dim;d+=32)s+=(qkv[h*dim+d]+bias[h*dim+d])*key[(h*capacity+p)*dim+d];
   for(int offset=16;offset;offset>>=1)s+=__shfl_down_sync(0xffffffff,s,offset);
   if(lane==0)scores[p]=s*rsqrtf((float)dim);}
 __syncthreads();
 float maximum=-3.402823466e+38F;
 for(int p=tid;p<n;p+=256)maximum=fmaxf(maximum,scores[p]);
 reduce[tid]=maximum;__syncthreads();
 for(int k=128;k;k>>=1){if(tid<k)reduce[tid]=fmaxf(reduce[tid],reduce[tid+k]);__syncthreads();}
 maximum=reduce[0];float total=0;
 for(int p=tid;p<n;p+=256){float s=expf(scores[p]-maximum);scores[p]=s;total+=s;}
 reduce[tid]=total;__syncthreads();
 for(int k=128;k;k>>=1){if(tid<k)reduce[tid]+=reduce[tid+k];__syncthreads();}
 total=reduce[0];
 for(int tile=0;tile<dim;tile+=32){
   int d=tile+lane;float s=0;
   if(d<dim)for(int p=warp;p<n;p+=8)s+=(scores[p]/total)*value[(h*capacity+p)*dim+d];
   reduce[tid]=s;__syncthreads();
   if(warp==0 && d<dim){float sum=0;for(int w=0;w<8;w++)sum+=reduce[w*32+lane];out[h*dim+d]=sum;}
   __syncthreads();
 }
}
'''


_FP16_SOURCE = r'''
#include <cuda_fp16.h>
extern "C" __global__ void layer_norm(__half* x, const __half* residual,
 const __half* bias, const __half* weight, const __half* shift, int width, float eps) {
  __shared__ float sums[256];
  int row=blockIdx.x, tid=threadIdx.x;
  float s=0;
  for(int c=tid;c<width;c+=256)
    s+=__half2float(x[row*width+c])+__half2float(residual[row*width+c])+__half2float(bias[c]);
  sums[tid]=s; __syncthreads();
  for(int n=128;n;n>>=1){if(tid<n)sums[tid]+=sums[tid+n];__syncthreads();}
  float mean=sums[0]/width; s=0;
  for(int c=tid;c<width;c+=256){
    float v=__half2float(x[row*width+c])+__half2float(residual[row*width+c])+__half2float(bias[c])-mean;
    s+=v*v;
  }
  sums[tid]=s; __syncthreads();
  for(int n=128;n;n>>=1){if(tid<n)sums[tid]+=sums[tid+n];__syncthreads();}
  float inv=rsqrtf(sums[0]/width+eps);
  for(int c=tid;c<width;c+=256){
    float v=__half2float(x[row*width+c])+__half2float(residual[row*width+c])+__half2float(bias[c]);
    x[row*width+c]=__float2half_rn((v-mean)*inv*__half2float(weight[c])+__half2float(shift[c]));
  }
}
extern "C" __global__ void embedding(__half* x,const __half* emb,const __half* pe,
 const __half* alpha,const int* state,int width) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;
 if(i<width)x[i]=__float2half_rn(__half2float(emb[state[0]*width+i])+
   __half2float(alpha[0])*__half2float(pe[state[2]*width+i]));
}
extern "C" __global__ void kv_write(const __half* qkv, const __half* bias,
 __half* key,__half* value,const int* state,int width,int dim,int capacity){
 int i=blockDim.x*blockIdx.x+threadIdx.x;
 if(i<width){int dest=(i/dim*capacity+state[1])*dim+i%dim;
 key[dest]=__float2half_rn(__half2float(qkv[width+i])+__half2float(bias[width+i]));
 value[dest]=__float2half_rn(__half2float(qkv[2*width+i])+__half2float(bias[2*width+i]));}
}
extern "C" __global__ void attention(const __half* qkv,const __half* bias,
 const __half* key,const __half* value,__half* out,const int* state,int dim,int capacity){
 extern __shared__ float scores[];
 __shared__ float reduce[256];
 int h=blockIdx.x,tid=threadIdx.x,n=state[1]+1;
 int lane=tid%32,warp=tid/32;
 for(int p=warp;p<n;p+=8){float s=0;
   for(int d=lane;d<dim;d+=32){
     float q=__half2float(__float2half_rn(__half2float(qkv[h*dim+d])+__half2float(bias[h*dim+d])));
     s+=q*__half2float(key[(h*capacity+p)*dim+d]);
   }
   for(int offset=16;offset;offset>>=1)s+=__shfl_down_sync(0xffffffff,s,offset);
   if(lane==0)scores[p]=s*rsqrtf((float)dim);}
 __syncthreads();
 float maximum=-3.402823466e+38F;
 for(int p=tid;p<n;p+=256)maximum=fmaxf(maximum,scores[p]);
 reduce[tid]=maximum;__syncthreads();
 for(int k=128;k;k>>=1){if(tid<k)reduce[tid]=fmaxf(reduce[tid],reduce[tid+k]);__syncthreads();}
 maximum=reduce[0];float total=0;
 for(int p=tid;p<n;p+=256){float s=expf(scores[p]-maximum);scores[p]=s;total+=s;}
 reduce[tid]=total;__syncthreads();
 for(int k=128;k;k>>=1){if(tid<k)reduce[tid]+=reduce[tid+k];__syncthreads();}
 total=reduce[0];
 for(int tile=0;tile<dim;tile+=32){
   int d=tile+lane;float s=0;
   if(d<dim)for(int p=warp;p<n;p+=8)s+=(scores[p]/total)*__half2float(value[(h*capacity+p)*dim+d]);
   reduce[tid]=s;__syncthreads();
   if(warp==0 && d<dim){float sum=0;for(int w=0;w<8;w++)sum+=reduce[w*32+lane];out[h*dim+d]=__float2half_rn(sum);}
   __syncthreads();
 }
}
'''


_SPLIT_KV_SOURCE = r'''
#include <cuda_fp16.h>
#if USE_FP16
typedef __half scalar_t;
__device__ float read_value(scalar_t v){return __half2float(v);}
__device__ scalar_t write_value(float v){return __float2half_rn(v);}
#else
typedef float scalar_t;
__device__ float read_value(scalar_t v){return v;}
__device__ scalar_t write_value(float v){return v;}
#endif
extern "C" __global__ void attention_split(const scalar_t* qkv,const scalar_t* bias,
 const scalar_t* key,const scalar_t* value,float* stats,float* partials,
 const int* state,int dim,int capacity,int chunk_size,int chunks){
 extern __shared__ float scores[];
 __shared__ float reduce[256];
 int h=blockIdx.x,chunk=blockIdx.y,tid=threadIdx.x;
 int start=chunk*chunk_size,n=state[1]+1;
 int count=min(chunk_size,max(0,n-start));
 int part=h*chunks+chunk,lane=tid%32,warp=tid/32;
 if(count==0){
   if(tid==0){stats[part*2]=-3.402823466e+38F;stats[part*2+1]=0;}
   for(int d=tid;d<dim;d+=256)partials[part*dim+d]=0;
   return;
 }
 for(int p=warp;p<count;p+=8){float s=0;
   for(int d=lane;d<dim;d+=32){
     float q=read_value(write_value(read_value(qkv[h*dim+d])+read_value(bias[h*dim+d])));
     s+=q*read_value(key[(h*capacity+start+p)*dim+d]);
   }
   for(int offset=16;offset;offset>>=1)s+=__shfl_down_sync(0xffffffff,s,offset);
   if(lane==0)scores[p]=s*rsqrtf((float)dim);
 }
 __syncthreads();
 float maximum=-3.402823466e+38F;
 for(int p=tid;p<count;p+=256)maximum=fmaxf(maximum,scores[p]);
 reduce[tid]=maximum;__syncthreads();
 for(int k=128;k;k>>=1){if(tid<k)reduce[tid]=fmaxf(reduce[tid],reduce[tid+k]);__syncthreads();}
 maximum=reduce[0];float total=0;
 for(int p=tid;p<count;p+=256){float s=expf(scores[p]-maximum);scores[p]=s;total+=s;}
 reduce[tid]=total;__syncthreads();
 for(int k=128;k;k>>=1){if(tid<k)reduce[tid]+=reduce[tid+k];__syncthreads();}
 if(tid==0){stats[part*2]=maximum;stats[part*2+1]=reduce[0];}
 __syncthreads();
 for(int tile=0;tile<dim;tile+=32){
   int d=tile+lane;float s=0;
   if(d<dim)for(int p=warp;p<count;p+=8)s+=scores[p]*read_value(value[(h*capacity+start+p)*dim+d]);
   reduce[tid]=s;__syncthreads();
   if(warp==0 && d<dim){float sum=0;for(int w=0;w<8;w++)sum+=reduce[w*32+lane];partials[part*dim+d]=sum;}
   __syncthreads();
 }
}
extern "C" __global__ void attention_merge(const float* stats,const float* partials,
 scalar_t* out,int dim,int chunks){
 extern __shared__ float factors[];
 __shared__ float reduce[256];
 int h=blockIdx.x,tid=threadIdx.x;
 float maximum=-3.402823466e+38F;
 for(int c=tid;c<chunks;c+=256){int part=h*chunks+c;
   if(stats[part*2+1]>0)maximum=fmaxf(maximum,stats[part*2]);
 }
 reduce[tid]=maximum;__syncthreads();
 for(int k=128;k;k>>=1){if(tid<k)reduce[tid]=fmaxf(reduce[tid],reduce[tid+k]);__syncthreads();}
 maximum=reduce[0];float total=0;
 for(int c=tid;c<chunks;c+=256){int part=h*chunks+c;
   float factor=stats[part*2+1]>0 ? expf(stats[part*2]-maximum) : 0;
   factors[c]=factor;total+=factor*stats[part*2+1];
 }
 reduce[tid]=total;__syncthreads();
 for(int k=128;k;k>>=1){if(tid<k)reduce[tid]+=reduce[tid+k];__syncthreads();}
 total=reduce[0];
 for(int d=tid;d<dim;d+=256){float sum=0;
   for(int c=0;c<chunks;c++)sum+=factors[c]*partials[(h*chunks+c)*dim+d];
   out[h*dim+d]=write_value(sum/total);
 }
}
'''


def _validate_attention(attention, chunk_size):
    if attention not in ("baseline", "split-kv"):
        raise ValueError("GPT attention must be baseline or split-kv")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size not in (256, 512):
        raise ValueError("GPT attention chunk size must be 256 or 512")


def _validate_prefill_query_chunk_size(chunk_size):
    if (isinstance(chunk_size, (bool, np.bool_))
            or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 0):
        raise ValueError("GPT prefill query chunk size must be a non-negative integer")


class CUDAGPT:
    def __init__(self, manifest, weights, capacity, use_graph=True, precision="fp32",
                 attention="baseline", attention_chunk_size=256, prefill_query_chunk_size=0):
        if precision not in ("fp32", "fp16"):
            raise ValueError("GPT precision must be fp32 or fp16")
        _validate_attention(attention, attention_chunk_size)
        _validate_prefill_query_chunk_size(prefill_query_chunk_size)
        self.prefill_query_chunk_size = int(prefill_query_chunk_size)
        self.attention, self.attention_chunk_size = attention, attention_chunk_size
        self.attention_chunks = (capacity+attention_chunk_size-1)//attention_chunk_size
        self.precision = precision
        self.dtype = cp.float16 if precision == "fp16" else cp.float32
        if any(weight.dtype != self.dtype for weight in weights.values()):
            raise ValueError("GPT weight dtype does not match execution precision")
        self.weight_manifest = manifest
        self.config = manifest["config"]
        self.weights = weights
        self.width = int(self.config["hidden_dim"])
        self.heads = int(self.config["heads"])
        self.layers = int(self.config["layers"])
        self.head_dim = self.width // self.heads
        self.epsilon = float(self.config["layer_norm_epsilon"])
        if capacity < 1 or self.head_dim > 256 or capacity > 10000:
            raise ValueError("Require KV capacity 1..10000 and head dimension <=256")
        self.capacity, self.use_graph = capacity, use_graph
        source = _FP16_SOURCE if precision == "fp16" else _SOURCE
        self.kernels = {name: cp.RawKernel(source, name, options=("--std=c++11", "--fmad=false"))
                        for name in ("layer_norm", "embedding", "kv_write", "attention")}
        if attention == "split-kv":
            split_source = f"#define USE_FP16 {int(precision == 'fp16')}\n" + _SPLIT_KV_SOURCE
            self.kernels.update({name: cp.RawKernel(split_source, name,
                options=("--std=c++11", "--fmad=false")) for name in ("attention_split", "attention_merge")})
        for kernel in self.kernels.values():
            kernel.compile()
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.blas = _GraphBLAS(self.stream, precision)
        self.keys = self.values = self.graph = self.workspace = None
        self.state = None
        self.length = self.text_length = 0

    @classmethod
    def load(cls, package, capacity=2048, use_graph=True, precision="fp32",
             attention="baseline", attention_chunk_size=256, prefill_query_chunk_size=0):
        if precision not in ("fp32", "fp16"):
            raise ValueError("GPT precision must be fp32 or fp16")
        _validate_attention(attention, attention_chunk_size)
        _validate_prefill_query_chunk_size(prefill_query_chunk_size)
        package = Path(package)
        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["format"] != "sakuratts-gpt-fp32-v1"
                or manifest["architecture"] != "gpt-sovits-ar-postnorm-relu"):
            raise ValueError("Unsupported GPT package format or architecture")
        path = package / manifest["weights"]["file"]
        if sha256_file(path) != manifest["weights"]["sha256"]:
            raise ValueError("GPT weight archive checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            validate_storage(manifest, archive.files)
            dtype = np.float16 if precision == "fp16" else np.float32
            weights = {}
            for name in archive.files:
                value = read_fp32(archive, manifest, name)
                if precision == "fp16":
                    # Check the expanded tensor before upload; no FP32 GPU
                    # copy or modified model package is needed.
                    if not np.isfinite(value).all() or np.any(np.abs(value) > np.finfo(np.float16).max):
                        raise ValueError(f"GPT weight cannot be represented as finite FP16: {name}")
                    value = np.ascontiguousarray(value, dtype=dtype)
                weights[name] = cp.asarray(value)
        cp.cuda.get_current_stream().synchronize()
        return cls(manifest, weights, capacity, use_graph, precision, attention,
                   attention_chunk_size, prefill_query_chunk_size)

    def _allocate_state(self):
        if self.keys is None:
            shape = (self.layers, self.heads, self.capacity, self.head_dim)
            self.keys, self.values = cp.empty(shape, self.dtype), cp.empty(shape, self.dtype)
            w = self.width
            self.workspace = {name: cp.empty(shape, cp.float32 if name == "logits" else self.dtype) for name, shape in {
                "x": (1, w), "qkv": (1, 3*w), "attention": (1, w),
                "mix": (1, w), "ffn": (1, 4*w), "out": (1, w),
                "logits": (1, self.config["vocab_size"])}.items()}
            if self.attention == "split-kv":
                self.workspace["attention_stats"] = cp.empty((self.heads, self.attention_chunks, 2), cp.float32)
                self.workspace["attention_partials"] = cp.empty((self.heads, self.attention_chunks, self.head_dim), cp.float32)
            self.state = cp.zeros(3, cp.int32)

    def release_request_state(self):
        self.stream.synchronize()
        self.graph = None
        self.keys = self.values = self.workspace = None
        self.state = None
        self.length = self.text_length = 0
        cp.get_default_memory_pool().free_all_blocks()

    def close(self):
        self.release_request_state()
        self.blas.close()
        self.weights.clear()
        cp.get_default_memory_pool().free_all_blocks()

    def _norm(self, x, residual, prefix, bias):
        self.kernels["layer_norm"]((x.shape[0],), (256,), (
            x, residual, bias, self.weights[prefix+".weight"], self.weights[prefix+".bias"],
            np.int32(self.width), np.float32(self.epsilon)))
        return x

    def prefill(self, phones, prompt, bert):
        phones, prompt, bert = np.asarray(phones), np.asarray(prompt), np.asarray(bert)
        if phones.dtype!=np.int64 or prompt.dtype!=np.int64 or bert.dtype!=np.float32:
            raise ValueError("Require int64 phones/prompt and FP32 BERT features")
        if phones.ndim != 2 or prompt.ndim != 2 or phones.shape[0] != 1 or prompt.shape[0] != 1:
            raise ValueError("Expected batch=1 phones and reference semantics")
        t, p = phones.shape[1], prompt.shape[1]
        if min(t,p) < 1 or t+p > self.capacity or max(t,p)>self.config["max_positions"]:
            raise ValueError("Empty sequence or GPT prefill capacity exceeded")
        if bert.shape != (1,t,self.config["bert_dim"]) or not np.isfinite(bert).all():
            raise ValueError("BERT features must be finite and align with all phones")
        if self.precision == "fp16" and np.any(np.abs(bert) > np.finfo(np.float16).max):
            raise ValueError("BERT features cannot be represented as finite FP16")
        if phones.min()<0 or phones.max()>=self.config["phoneme_vocab_size"] or prompt.min()<0 or prompt.max()>=self.config["vocab_size"]:
            raise ValueError("Phone or semantic token outside model vocabulary")
        self._allocate_state()
        self.text_length, self.length = t, t+p
        w = self.weights
        with self.stream:
            if self.precision == "fp16":
                result = cp.asnumpy(self._prefill_fp16(phones, prompt, bert))
                self.stream.synchronize()
                return result
            text = w["text_embedding"][cp.asarray(phones[0])] + cp.asarray(bert[0]) @ w["bert.weight"].T + w["bert.bias"]
            text += w["text_alpha"] * w["position_encoding"][:t]
            audio = w["audio_embedding"][cp.asarray(prompt[0])] + w["audio_alpha"] * w["position_encoding"][:p]
            x = cp.concatenate((text,audio))
            if not self.prefill_query_chunk_size:
                allowed = np.zeros((t+p,t+p),bool)
                allowed[:,:t] = True
                allowed[t:,t:] = np.tril(np.ones((p,p),bool))
                mask = cp.asarray(allowed)
            for layer in range(self.layers):
                pre = f"layers.{layer}."
                qkv = x @ w[pre+"qkv.weight"].T + w[pre+"qkv.bias"]
                q,k,v = [a.reshape(-1,self.heads,self.head_dim).transpose(1,0,2) for a in cp.split(qkv,3,axis=-1)]
                self.keys[layer,:,:t+p] = k
                self.values[layer,:,:t+p] = v
                if self.prefill_query_chunk_size:
                    attended = self._prefill_attention_chunked(q, k, v, t)
                else:
                    scores = (q @ k.transpose(0,2,1)) * np.float32(self.head_dim**-0.5)
                    scores = cp.where(mask, scores, -cp.inf)
                    scores -= cp.max(scores, axis=-1, keepdims=True)
                    cp.exp(scores, out=scores)
                    scores /= cp.sum(scores,axis=-1,keepdims=True)
                    attended = (scores @ v).transpose(1,0,2).reshape(-1,self.width)
                mixed = attended @ w[pre+"attention_output.weight"].T
                mixed = self._norm(mixed,x,pre+"norm1",w[pre+"attention_output.bias"])
                ffn = cp.maximum(mixed @ w[pre+"ffn_in.weight"].T + w[pre+"ffn_in.bias"],0)
                x = self._norm(ffn @ w[pre+"ffn_out.weight"].T,mixed,pre+"norm2",w[pre+"ffn_out.bias"])
            logits = x[-1:] @ w["output.weight"].T
            result = cp.asnumpy(logits)
        self.stream.synchronize()
        return result

    def _linear_fp16(self, x, weight, *, output_dtype=None):
        out = cp.empty((x.shape[0], weight.shape[0]), output_dtype or self.dtype)
        self.blas.linear(x, weight, out)
        return out

    def _prefill_attention_chunked(self, q, k, v, text_length):
        """Bound score/mask storage by query rows while retaining every key.

        Global positions preserve bidirectional text and causal audio across
        chunk boundaries. FP16 uses the same FP32 QK/softmax accumulation and
        FP16 probabilities as the full prefill path.
        """
        length = q.shape[1]
        attended = cp.empty_like(v)
        key_positions = cp.arange(length)[None, :]
        for start in range(0, length, self.prefill_query_chunk_size):
            stop = min(start + self.prefill_query_chunk_size, length)
            query_positions = cp.arange(start, stop)[:, None]
            allowed = ((key_positions < text_length)
                       | ((query_positions >= text_length) & (key_positions <= query_positions)))
            if self.precision == "fp16":
                query = cp.ascontiguousarray(q[:, start:stop])
                scores = cp.empty((self.heads, stop-start, length), cp.float32)
                self.blas._gemm_fp16(query, k, scores, transpose_y=True)
                scores *= np.float32(self.head_dim**-0.5)
            else:
                scores = (q[:, start:stop] @ k.transpose(0, 2, 1)) * np.float32(self.head_dim**-0.5)
            scores = cp.where(allowed, scores, -cp.inf)
            scores -= cp.max(scores, axis=-1, keepdims=True)
            cp.exp(scores, out=scores)
            scores /= cp.sum(scores, axis=-1, keepdims=True, dtype=cp.float32)
            if self.precision == "fp16":
                block = cp.empty((self.heads, stop-start, self.head_dim), self.dtype)
                self.blas._gemm_fp16(scores.astype(self.dtype), v, block)
                attended[:, start:stop] = block
                del query, block
            else:
                attended[:, start:stop] = scores @ v
            # Release the previous block before allocating the next one.
            del scores, allowed
        return cp.ascontiguousarray(attended.transpose(1, 0, 2).reshape(-1, self.width))

    def _prefill_fp16(self, phones, prompt, bert):
        """Mixed-precision prefix using the same resident weights as decode.

        QK produces FP32 scores. Softmax reductions stay FP32; probabilities
        are rounded to FP16 for the AV GEMM, whose accumulation remains FP32.
        LayerNorm and decode attention also accumulate in FP32.
        """
        t, p = phones.shape[1], prompt.shape[1]
        w = self.weights
        bert_gpu = cp.asarray(bert[0], dtype=self.dtype, order="C")
        text = w["text_embedding"][cp.asarray(phones[0])] + self._linear_fp16(bert_gpu, w["bert.weight"]) + w["bert.bias"]
        text += w["text_alpha"] * w["position_encoding"][:t]
        audio = w["audio_embedding"][cp.asarray(prompt[0])] + w["audio_alpha"] * w["position_encoding"][:p]
        x = cp.concatenate((text, audio))
        if not self.prefill_query_chunk_size:
            allowed = np.zeros((t+p, t+p), bool)
            allowed[:, :t] = True
            allowed[t:, t:] = np.tril(np.ones((p, p), bool))
            mask = cp.asarray(allowed)
        for layer in range(self.layers):
            pre = f"layers.{layer}."
            qkv = self._linear_fp16(x, w[pre+"qkv.weight"]) + w[pre+"qkv.bias"]
            q, k, v = [cp.ascontiguousarray(a.reshape(-1, self.heads, self.head_dim).transpose(1, 0, 2))
                       for a in cp.split(qkv, 3, axis=-1)]
            self.keys[layer, :, :t+p] = k
            self.values[layer, :, :t+p] = v
            if self.prefill_query_chunk_size:
                attended = self._prefill_attention_chunked(q, k, v, t)
            else:
                scores = cp.empty((self.heads, t+p, t+p), cp.float32)
                self.blas._gemm_fp16(q, k, scores, transpose_y=True)
                scores *= np.float32(self.head_dim**-0.5)
                scores = cp.where(mask, scores, -cp.inf)
                scores -= cp.max(scores, axis=-1, keepdims=True)
                cp.exp(scores, out=scores)
                scores /= cp.sum(scores, axis=-1, keepdims=True, dtype=cp.float32)
                attended = cp.empty_like(v)
                self.blas._gemm_fp16(scores.astype(self.dtype), v, attended)
                attended = cp.ascontiguousarray(attended.transpose(1, 0, 2).reshape(-1, self.width))
            mixed = self._linear_fp16(attended, w[pre+"attention_output.weight"])
            mixed = self._norm(mixed, x, pre+"norm1", w[pre+"attention_output.bias"])
            ffn = cp.maximum(self._linear_fp16(mixed, w[pre+"ffn_in.weight"]) + w[pre+"ffn_in.bias"], 0)
            x = self._norm(self._linear_fp16(ffn, w[pre+"ffn_out.weight"]), mixed,
                           pre+"norm2", w[pre+"ffn_out.bias"])
        return self._linear_fp16(x[-1:], w["output.weight"], output_dtype=cp.float32)

    def _decode_graph_body(self):
        w,b = self.weights,self.workspace
        self.kernels["embedding"](((self.width+255)//256,), (256,), (
            b["x"],w["audio_embedding"],w["position_encoding"],w["audio_alpha"],self.state,np.int32(self.width)))
        for layer in range(self.layers):
            pre=f"layers.{layer}."
            self.blas.linear(b["x"],w[pre+"qkv.weight"],b["qkv"])
            self.kernels["kv_write"](((self.width+255)//256,), (256,), (
                b["qkv"],w[pre+"qkv.bias"],self.keys[layer],self.values[layer],self.state,
                np.int32(self.width),np.int32(self.head_dim),np.int32(self.capacity)))
            if self.attention == "split-kv":
                self.kernels["attention_split"]((self.heads, self.attention_chunks), (256,), (
                    b["qkv"],w[pre+"qkv.bias"],self.keys[layer],self.values[layer],
                    b["attention_stats"],b["attention_partials"],self.state,
                    np.int32(self.head_dim),np.int32(self.capacity),
                    np.int32(self.attention_chunk_size),np.int32(self.attention_chunks)),
                    shared_mem=self.attention_chunk_size*4)
                self.kernels["attention_merge"]((self.heads,), (256,), (
                    b["attention_stats"],b["attention_partials"],b["attention"],
                    np.int32(self.head_dim),np.int32(self.attention_chunks)),
                    shared_mem=self.attention_chunks*4)
            else:
                self.kernels["attention"]((self.heads,), (256,), (
                    b["qkv"],w[pre+"qkv.bias"],self.keys[layer],self.values[layer],b["attention"],self.state,
                    np.int32(self.head_dim),np.int32(self.capacity)),shared_mem=self.capacity*4)
            self.blas.linear(b["attention"],w[pre+"attention_output.weight"],b["mix"])
            self._norm(b["mix"],b["x"],pre+"norm1",w[pre+"attention_output.bias"])
            self.blas.linear(b["mix"],w[pre+"ffn_in.weight"],b["ffn"])
            cp.add(b["ffn"],w[pre+"ffn_in.bias"],out=b["ffn"])
            cp.maximum(b["ffn"],0,out=b["ffn"])
            self.blas.linear(b["ffn"],w[pre+"ffn_out.weight"],b["x"])
            self._norm(b["x"],b["mix"],pre+"norm2",w[pre+"ffn_out.bias"])
        self.blas.linear(b["x"],w["output.weight"],b["logits"])

    def decode(self, token):
        if not isinstance(token,(int,np.integer)):
            raise ValueError("Semantic token must be an integer")
        if not self.length:
            raise RuntimeError("Call prefill before decode")
        if self.length>=self.capacity or self.length-self.text_length>=self.config["max_positions"]:
            raise ValueError("GPT decode capacity exceeded; increase capacity, do not truncate text")
        if token<0 or token>=self.config["vocab_size"]:
            raise ValueError("Semantic token outside model vocabulary")
        with self.stream:
            self.state.set(np.array((token,self.length,self.length-self.text_length),np.int32))
            if self.use_graph:
                if self.graph is None:
                    self._decode_graph_body()
                    self.stream.synchronize()
                    self.stream.begin_capture()
                    try:
                        self._decode_graph_body()
                        self.graph=self.stream.end_capture()
                    except BaseException:
                        # CUDA invalidates a failed capture. End it before
                        # request cleanup attempts stream synchronization.
                        try:
                            self.stream.end_capture()
                        except Exception:
                            pass
                        self.graph=None
                        raise
                self.graph.launch(self.stream)
            else:
                self._decode_graph_body()
            result=cp.asnumpy(self.workspace["logits"])
        self.stream.synchronize()
        self.length+=1
        return result
