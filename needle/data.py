# %% Import Libs

import gzip
import struct

import numpy as np

from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any, Tuple

# %% Transformations

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        return np.flip(img, axis=1) if flip_img else img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img: np.ndarray):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Returns:
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        h, w, _ = img.shape
        pad_widths = ((self.padding,self.padding), (self.padding,self.padding), (0,0))
        img_padded = np.pad(img, pad_widths, 'constant', constant_values=0)
        img_rolled = np.roll(img_padded, shift=(-shift_x,-shift_y,0), axis=(0,1,2))
        return img_rolled[self.padding:self.padding+h,self.padding:self.padding+w,:]
        ### END YOUR SOLUTION

# %% Dataset Base Class

class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x

# %% DataLoader Base Class

class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)),
                range(batch_size, len(dataset), batch_size)
            )

    def __iter__(self):
        # called at the start of iteration
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        # called to grab the next mini-batch
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

# %% MNIST Dataset

DTYPE = {
    8 : (np.uint8,   'B'),
    9 : (np.int8,    'b'),
    11: (np.int16,   'h'),
    12: (np.int32,   'i'),
    13: (np.float32, 'f'),
    14: (np.float64, 'd')
}

def parse_idx_file(idx_bin_data: bytes, dtype=None):
    '''
    magic number          32-bit integer (00 dtype n_dim)
    size in dimension 0   32-bit integer
    size in dimension 1   32-bit integer
    size in dimension 2   32-bit integer
    .....
    size in dimension N   32-bit integer
    data
    '''
    _dtype, fmt_char = DTYPE[idx_bin_data[2]]
    dtype = dtype if dtype is not None else _dtype
    n_dim = idx_bin_data[3]
    shape = struct.unpack_from(f">{n_dim}i", idx_bin_data, offset=4)
    n_data = np.prod(shape)
    data = struct.unpack_from(f">{n_data}{fmt_char}", idx_bin_data, offset=4*(n_dim+1))
    return np.array(data, dtype).reshape(shape)

def parse_mnist(image_filename, label_filename) -> Tuple[np.ndarray, np.ndarray]:
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    # load data
    with gzip.open(image_filename) as f:
        X = parse_idx_file(f.read(), np.float32)
    with gzip.open(label_filename) as f:
        y = parse_idx_file(f.read())
    # process data
    X = X.reshape(X.shape[0], -1)
    X = (X - X.min()) / (X.max() - X.min())
    return X, y

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.X, self.y = parse_mnist(image_filename, label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        ### BEGIN YOUR SOLUTION
        return self.apply_transforms(self.X[idx,:].reshape((28,28,1))), self.y[idx]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION

# %% NDArray Dataset

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
