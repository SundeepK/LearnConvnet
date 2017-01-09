import json
import numpy
import io
import base64


class NumpyCompressedArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            output = io.BytesIO()
            numpy.savez_compressed(output, numpy_array=obj)
            return base64.b64encode(output.getvalue())
        elif isinstance(obj, numpy.generic):
            return obj.item()
        return json.JSONEncoder.default(self, obj)
