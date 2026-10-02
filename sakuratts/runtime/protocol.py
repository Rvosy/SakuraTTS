"""Length-framed NumPy arrays for the local acoustic worker, without pickle."""

import json
import struct

import numpy as np


def _read(stream, size):
    parts = bytearray(size)
    view = memoryview(parts)
    while view:
        count = stream.readinto(view)
        if not count:
            raise EOFError("Acoustic worker closed its output before completing a message")
        view = view[count:]
    return parts


def write_message(stream, metadata, arrays=None):
    arrays = {name: np.asarray(value,order="C") for name,value in (arrays or {}).items()}
    header = dict(metadata)
    header["arrays"] = {name:{"shape":list(value.shape),"dtype":value.dtype.str,"bytes":value.nbytes}
                        for name,value in arrays.items()}
    encoded = json.dumps(header,ensure_ascii=False).encode("utf-8")
    stream.write(struct.pack("<I",len(encoded)))
    stream.write(encoded)
    for value in arrays.values():
        stream.write(memoryview(value).cast("B"))
    stream.flush()


def read_message(stream):
    size = struct.unpack("<I",_read(stream,4))[0]
    if not 0<size<=1024*1024:
        raise ValueError("Invalid acoustic message header length")
    header = json.loads(_read(stream,size).decode("utf-8"))
    arrays = {}
    for name,spec in header.pop("arrays").items():
        dtype = np.dtype(spec["dtype"])
        if dtype not in (np.dtype("float32"),np.dtype("int64")):
            raise ValueError("Unsupported acoustic transport dtype")
        shape = tuple(spec["shape"])
        if any(not isinstance(v,int) or v<0 for v in shape):
            raise ValueError("Invalid acoustic transport shape")
        size = int(np.prod(shape,dtype=np.int64))*dtype.itemsize
        if size!=spec["bytes"] or not 0<=size<=1024**3:
            raise ValueError("Invalid acoustic transport payload length")
        arrays[name] = np.frombuffer(_read(stream,size),dtype=dtype).reshape(shape)
    return header,arrays
