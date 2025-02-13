#!/usr/bin/env python3
"""
Inner product Python wrapper, including gradients. By making this the main
entry point to _inner_product_impl.so, we ensure that appropriate gradient ops
are *always* registered whenever we want to use the inner_product op.

.. moduleauthor:: David Stutz
"""

import os.path as osp

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

module_dir = osp.dirname(osp.abspath(__file__))
inner_product_module = tf.load_op_library(
    osp.join(module_dir, '_inner_product_impl.so'))
inner_product = inner_product_module.inner_product
inner_product_grad = inner_product_module.inner_product_grad

__all__ = ['inner_product', 'inner_product_grad']


@ops.RegisterGradient("InnerProduct")
def _inner_product_grad_cc(op, grad):
    """
    The gradient for `inner_product` using the operation implemented in C++.

    :param op: `inner_product` `Operation` that we are differentiating, which
        we can use to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `inner_product` op.
    :return: gradients with respect to the input of `inner_product`.
    """

    return inner_product_grad(grad, op.inputs[0], op.inputs[1])


# uncomment this and comment the corresponding line above to use the Python
# implementation of the inner product gradient
# @ops.RegisterGradient("InnerProduct")
def _inner_product_grad(op, grad):
    """
    The gradients for `inner_product`.

    :param op: `inner_product` `Operation` that we are differentiating, which
        we can use to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `inner_product` op.
    :return: gradients with respect to the input of `inner_product`.
    """

    input_tensor = op.inputs[0]
    weight_tensor = op.inputs[1]
    input_rows = array_ops.shape(input_tensor)[0]
    output_rows = array_ops.shape(weight_tensor)[0]

    grad_input = tf.matmul(tf.transpose(grad), weight_tensor)
    grad_weights = tf.multiply(
        tf.transpose(grad),
        tf.reshape(
            tf.tile(tf.reshape(input_tensor, [input_rows]), [output_rows]),
            [output_rows, -1]))

    return [tf.transpose(grad_input), grad_weights]
