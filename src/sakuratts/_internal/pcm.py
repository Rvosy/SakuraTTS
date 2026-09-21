"""PCM transport for the control process without numerical-library imports."""

from array import array
import sys


def pcm_from_s16le(data):
    pcm = array("h")
    pcm.frombytes(data)
    if sys.byteorder != "little":
        pcm.byteswap()
    return pcm


def pcm_s16le_bytes(pcm):
    if isinstance(pcm, array):
        if pcm.typecode != "h" or pcm.itemsize != 2:
            raise ValueError("PCM must contain signed 16-bit samples")
        if sys.byteorder != "little":
            pcm = array("h", pcm)
            pcm.byteswap()
        return pcm.tobytes()
    # Preserve the original conversion for NumPy arrays from direct inference.
    return pcm.astype("<i2", copy=False).tobytes()
