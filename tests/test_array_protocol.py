"""The offline ORT worker transports full arrays and rejects broken messages."""

import io
import json
from pathlib import Path
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from sakuratts._internal.protocol import read_message,write_message


class ArrayProtocolTests(unittest.TestCase):
    def test_noncontiguous_arrays_roundtrip_without_pickle_or_precision_changes(self):
        source={"phones":np.arange(15,dtype=np.int64).reshape(3,5)[:,::2],
                "audio":np.linspace(-1,1,4101,dtype=np.float32)}
        stream=io.BytesIO()
        write_message(stream,{"reference":"中性"},source)
        stream.seek(0)
        meta,actual=read_message(stream)
        self.assertEqual(meta,{"reference":"中性"})
        for key in source:
            np.testing.assert_array_equal(actual[key],source[key])
            self.assertEqual(actual[key].dtype,source[key].dtype)

    def test_truncated_body_raises_instead_of_returning_partial_waveform(self):
        stream=io.BytesIO()
        write_message(stream,{}, {"audio":np.ones(10,np.float32)})
        with self.assertRaises(EOFError):
            read_message(io.BytesIO(stream.getvalue()[:-1]))

    def test_scalar_shape_is_preserved(self):
        stream=io.BytesIO()
        write_message(stream,{}, {"scale":np.asarray(.5,np.float32)})
        stream.seek(0)
        _,arrays=read_message(stream)
        self.assertEqual(arrays["scale"].shape,())

    def test_object_dtype_is_rejected_before_decoding_payload(self):
        header=json.dumps({"arrays":{"bad":{"shape":[1],"dtype":"O","bytes":8}}}).encode()
        with self.assertRaisesRegex(ValueError,"dtype"):
            read_message(io.BytesIO(struct.pack("<I",len(header))+header))


if __name__=="__main__":
    unittest.main()
