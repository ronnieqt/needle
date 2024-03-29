# %% Import Libs

import os
import gzip
import struct
import pickle

import numpy as np

from pathlib import Path

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
        self.batch_idx = 0
        if self.shuffle:
            self.ordering = np.array_split(
                np.random.permutation(len(self.dataset)),
                range(self.batch_size, len(self.dataset), self.batch_size)
            )
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        # called to grab the next mini-batch
        ### BEGIN YOUR SOLUTION
        if self.batch_idx < len(self.ordering):
            idx = self.ordering[self.batch_idx]
            batch = map(list, zip(*[self.dataset[i] for i in idx]))
            self.batch_idx += 1
            return tuple(Tensor(data, requires_grad=False) for data in batch)
        else:
            raise StopIteration
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

    def __getitem__(self, idx: Union[int, slice]) -> Tuple[np.ndarray, np.ndarray]:
        ### BEGIN YOUR SOLUTION
        X_transformed = \
            np.apply_along_axis(
                func1d=lambda x: self.apply_transforms(x.reshape((28,28,1))),
                axis=1, arr=self.X[idx,:]
            ) \
            if isinstance(idx, slice) else \
            self.apply_transforms(self.X[idx,:].reshape((28,28,1)))
        return X_transformed, self.y[idx]
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

# %% CIFAR10 Dataset

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[float] = 0.5,  # TODO: check when to use this p?
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        base_path = Path(base_folder)
        if train:
            images, labels = [], []
            for i in range(1,6):
                data = unpickle(base_path / f"data_batch_{i}")
                images.append(data[b"data"])
                labels.append(data[b"labels"])
            self.X = np.vstack(images)
            self.y = np.concatenate(labels)
        else:
            data = unpickle(base_path / "test_batch")
            self.X = data[b"data"]
            self.y = np.array(data[b"labels"])
        self.X = self.X.astype("float32") / 255
        self.out_shape = (3, 32, 32)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        X_transformed = \
            np.apply_along_axis(
                func1d=lambda x: self.apply_transforms(x.reshape(self.out_shape)),
                axis=1, arr=self.X[index,:]
            ) \
            if isinstance(index, slice) else \
            self.apply_transforms(self.X[index,:].reshape(self.out_shape))
        return X_transformed, self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION

# %% Dictionary: word -> int

class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word: str) -> int:
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION

# %% Corpus

class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        ids = []
        with open(path, 'r') as f:
            n_lines = 0
            while True:
                # read a line
                line = f.readline()
                if line == "":
                    break
                # process the line
                words = line.strip().split()
                for word in words:
                    ids.append(self.dictionary.add_word(word))
                ids.append(self.dictionary.add_word("<eos>"))
                n_lines += 1
                # break if max_lines reached
                if max_lines is not None and n_lines >= max_lines:
                    break
        return ids
        ### END YOUR SOLUTION

# %% sequence to batch

def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    n_batch = len(data) // batch_size
    return (
        np.array(data[:n_batch*batch_size], dtype)
        .reshape((n_batch, batch_size), order='F')
    )
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    seq_len = min(bptt, batches.shape[0]-1-i)
    X = batches[i:i+seq_len, :]
    y = batches[i+1:i+1+seq_len,:]
    return Tensor(X, device=device, dtype=dtype, requires_grad=False), \
           Tensor(y.flatten(), device=device, dtype=dtype, requires_grad=False)
    ### END YOUR SOLUTION
