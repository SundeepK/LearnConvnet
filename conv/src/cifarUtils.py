import pickle
import numpy
import base64
import io

def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = numpy.array(Y)
        return X, Y

def decode_numpy(base64_encoded_numpy_arr):
    output = io.BytesIO(base64.b64decode(base64_encoded_numpy_arr))
    output.seek(0)
    array_ = numpy.load(output)['numpy_array']
    return array_
