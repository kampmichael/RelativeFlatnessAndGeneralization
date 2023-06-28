from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
import functorch

### CutOut regularization, implementation from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


### Conv2D to FC layer, adapted implementation from https://gist.github.com/vvolhejn/e265665c65d3df37e381316bf57b8421
"""
The function `torch_conv_layer_to_affine` takes a `torch.nn.Conv2d` layer `conv`
and produces an equivalent `torch.nn.Linear` layer `fc`.
Specifically, this means that the following holds for `x` of a valid shape:
    torch.flatten(conv(x)) == fc(torch.flatten(x))
Or equivalently:
    conv(x) == fc(torch.flatten(x)).reshape(conv(x).shape)
allowing of course for some floating-point error.
"""
def torch_conv_layer_to_affine(
    conv: torch.nn.Parameter, padding: Tuple[int, int], stride: Tuple[int, int], input_size: Tuple[int, int]
) -> torch.nn.Parameter:
    w, h = input_size
    kernel_size = (conv.shape[-2], conv.shape[-1])
    out_channels = conv.shape[0]
    in_channels = conv.shape[1]
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)

    # Formula from the Torch docs:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    output_size = [
        (input_size[i] + 2 * padding[i] - kernel_size[i]) // stride[i]
        + 1
        for i in [0, 1]
    ]

    in_shape = (in_channels, w, h)
    out_shape = (out_channels, output_size[0], output_size[1])

    # we do not need bias here
    fc = nn.Linear(in_features=np.product(in_shape), out_features=np.product(out_shape), bias=False)
    fc.weight.data.fill_(0.0)

    # Output coordinates
    for xo, yo in range2d(output_size[0], output_size[1]):
        # The upper-left corner of the filter in the input tensor
        xi0 = -padding[0] + stride[0] * xo
        yi0 = -padding[1] + stride[1] * yo

        # Position within the filter
        with torch.no_grad():
            for xd, yd in range2d(kernel_size[0], kernel_size[1]):
                # Output channel
                for co in range(out_channels):
                    # do not need bias
                    # fc.bias[enc_tuple((co, xo, yo), out_shape)] = conv.bias[co]
                    for ci in range(in_channels):
                        # Make sure we are within the input image (and not in the padding)
                        if 0 <= xi0 + xd < w and 0 <= yi0 + yd < h:
                            cw = conv[co, ci, xd, yd]
                            # Flatten the weight position to 1d in "canonical ordering",
                            # i.e. guaranteeing that:
                            # FC(img.reshape(-1)) == Conv(img).reshape(-1)
                            fc.weight[
                                enc_tuple((co, xo, yo), out_shape),
                                enc_tuple((ci, xi0 + xd, yi0 + yd), in_shape),
                            ] = cw

    return fc

def range2d(to_a, to_b):
    for a in range(to_a):
        for b in range(to_b):
            yield a, b

def enc_tuple(tup: Tuple, shape: Tuple) -> int:
    res = 0
    coef = 1
    for i in reversed(range(len(shape))):
        assert tup[i] < shape[i]
        res += coef * tup[i]
        coef *= shape[i]

    return res

def dec_tuple(x: int, shape: Tuple) -> Tuple:
    res = []
    for i in reversed(range(len(shape))):
        res.append(x % shape[i])
        x //= shape[i]

    return tuple(reversed(res))


if __name__ == "__main__":
    def test_tuple_encoding():
        x = enc_tuple((3, 2, 1), (5, 6, 7))
        assert dec_tuple(x, (5, 6, 7)) == (3, 2, 1)
        print("Tuple encoding ok")

    def test_layer_conversion():
        for stride in [1, 2]:
            for padding in [0, 1, 2]:
                for filter_size in [3, 4]:
                    img = torch.rand((1, 2, 6, 7))
                    conv = nn.Conv2d(2, 5, filter_size, stride=stride, padding=padding, bias=False)
                    fc = torch_conv_layer_to_affine(next(conv.parameters()), padding, stride, img.shape[2:])

                    # Also checks that our encoding flattens the inputs/outputs such that
                    # FC(flatten(img)) == flatten(Conv(img))
                    res1 = fc(img.reshape((-1))).reshape(conv(img).shape)
                    res2 = conv(img)
                    worst_error = (res1 - res2).max()

                    print("Output shape", res2.shape, "Worst error: ", float(worst_error))
                    assert worst_error <= 1.0e-6

        print("Layer conversion ok")

    test_tuple_encoding()
    test_layer_conversion()