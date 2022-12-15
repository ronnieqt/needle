"""Core data structures."""

# %% Import Libs

from typing import List, Set, Dict, Optional, NamedTuple, Tuple, Union
from collections import namedtuple

import numpy

import needle
from needle import init

from .backend_selection import Device, array_api, NDArray, default_device, BACKEND

# %% Global Variables

# needle version
LAZY_MODE = False
TENSOR_COUNTER = 0

# %% Operator

class Op:
    """Operator definition."""

    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: Tuple[NDArray]):
        """Calculate forward pass of operator.

        Parameters
        ----------
        input: np.ndarray (raw data objects, not Tensor objects)
            A list of input arrays to the function

        Returns
        -------
        output: nd.array
            Array output of the operation

        """
        raise NotImplementedError()

    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Union["Value", Tuple["Value"]]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: Value or Tuple[Value]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        """ Convenience method to always return a tuple from gradient call"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)


class TensorOp(Op):
    """ Op class specialized to output tensors, will be alternate subclasses for other structures """

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)  # this function creates the computational graph


class TensorTupleOp(Op):
    """Op class specialized to output TensorTuple"""

    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)

# %% Value (Node) and Tensor

class Value:
    """A value in the computational graph.

    This class helps us to represent a NODE in the computational graph.
    """

    # trace of computational graph
    # the following two fields capture how does the Value get computed
    op: Optional[Op]       # operation: how this Value is computed
    inputs: List["Value"]  # inputs: a list of Values that are fed into the above op
    # The following fields are cached fields for dynamic computation
    cached_data: NDArray   # used to hold the data
    requires_grad: bool    # indicates whether we want to compute gradient w.r.t. this Value or not

    def realize_cached_data(self) -> NDArray:                                   # IMPORTANT
        """Run compute to realize the cached data"""
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data
        self.cached_data = self.op.compute(*[x.realize_cached_data() for x in self.inputs])
        # assert self.cached_data.dtype == self.inputs[0].dtype, \
        #     f"OP: {type(self.op).__name__}; " \
        #     f"inputs: {', '.join([str(i.dtype) for i in self.inputs])}; " \
        #     f"output: {self.cached_data.dtype}"
        return self.cached_data

    def is_leaf(self) -> bool:
        return self.op is None

    def __del__(self) -> None:
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],
        *,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None
    ) -> None:
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data, *, requires_grad=False) -> "Value":
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]) -> "Value":
        '''Create a variable and attach it to the computational graph.'''
        value = cls.__new__(cls)
        value._init(op, inputs)
        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value

    def numpy(self) -> numpy.ndarray:
        data = self.realize_cached_data()
        if array_api is numpy:
            return data
        return data.numpy() if not isinstance(data, tuple) else [x.numpy() for x in data]


class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = "float32" if (BACKEND == "nd") else array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else default_device()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]) -> "Tensor":                # IMPORTANT
        '''Create a tensor and attach it to the computational graph.'''
        # we are not using __init__(...) here because, we have already
        # overloaded this function for the user-facing construction that
        # takes in an array and does certain operations
        tensor = Tensor.__new__(Tensor)   # construct a Tensor object
        tensor._init(op, inputs)          # populate the fields of the tensor
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            # carry out real computation(s) in the computational graph
            tensor.realize_cached_data()  # populate the cache_data field(s) in Value(s)
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False) -> "Tensor":
        '''Create a tensor and detach it from the computational graph.'''
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            op=None,
            inputs=[],  # detached from the computational graph
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self) -> "Tensor":
        '''Create a detached tensor that does not require grad (a shortcut for detach())'''
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, f"{value.dtype} {self.dtype}"
        self.cached_data = value.realize_cached_data()

    def detach(self) -> "Tensor":
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return default_device()
        return data.device

    def backward(self, out_grad=None) -> None:
        out_grad = out_grad if out_grad else \
                   init.ones(*self.shape, dtype=self.dtype, device=self.device)
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        return "needle.Tensor(\n" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def __add__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, other)  # calls TensorOp.__call__
        else:
            return needle.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseMul()(self, other)
        else:
            return needle.ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            raise NotImplementedError()
        else:
            return needle.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, needle.ops.Negate()(other))
        else:
            return needle.ops.AddScalar(-other)(self)

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(needle.ops.Negate()(self), other)
        else:
            return needle.ops.AddScalar(other)(-self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseDiv()(self, other)
        else:
            return needle.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return needle.ops.MatMul()(self, other)

    def matmul(self, other):
        return needle.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return needle.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return needle.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return needle.ops.Reshape(shape)(self)

    def __neg__(self):
        return needle.ops.Negate()(self)

    def transpose(self, axes=None):
        return needle.ops.Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__


class TensorTuple(Value):
    """Represent a tuple of tensors.

    To keep things simple, we do not support nested tuples.
    """

    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return needle.ops.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "needle.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return needle.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return TensorTuple.make_const(self.realize_cached_data())

# %% Helpers

def compute_gradient_of_variables(output_tensor: Tensor, out_grad: Tensor) -> None:
    """Take gradient of output node w.r.t. each node (Value) in the computation graph.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order: List[Tensor]
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    ### BEGIN YOUR SOLUTION
    for vj in reverse_topo_order:
        # vj.grad: vj_adjoint
        vj.grad = sum_node_list(node_to_output_grads_list[vj])
        if vj.is_leaf(): continue  # leaf node does not have any op or input
        partial_adjoints: Tuple[Tensor]
        partial_adjoints = vj.op.gradient_as_tuple(out_grad=vj.grad, node=vj)
        for vi, partial_adjoint in zip(vj.inputs, partial_adjoints):
            if vi in node_to_output_grads_list:
                node_to_output_grads_list[vi].append(partial_adjoint)
            else:
                node_to_output_grads_list[vi] = [partial_adjoint]
    ### END YOUR SOLUTION


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    ### BEGIN YOUR SOLUTION
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order
    ### END YOUR SOLUTION


def topo_sort_dfs(node: Value, visited: Set[Value], topo_order: List[Value]) -> None:
    """Post-order DFS"""
    ### BEGIN YOUR SOLUTION
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)
    ### END YOUR SOLUTION


##############################
####### Helper Methods #######
##############################


def sum_node_list(node_list: List[Tensor]) -> Tensor:
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)

# %% Notes

# About __new__():
# - __new__() is a static method of the object class.
# - When you create an instance of a class, python first
#       calls the __new__() method to create the object, and then
#       calls the __init__() method to initialize the object's attributes.
# - Signature: MyClass.__new__(class, *args, **kwargs)
#       The 1st argument is the class of the new object you want to create.
# - Python will not call the __init__() method automatically
#       if you explicitly create a new object using the __new__() method.
# - In practice, you use the __new__() method when you want to
#       tweak the object at the instantiated (creation) time.
