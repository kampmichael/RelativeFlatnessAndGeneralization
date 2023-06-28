
# modified from 
# https://docs.monai.io/en/stable/_modules/monai/networks/nets/varautoencoder.html

## Autoencoder with linear layers that connect the down- and up-sampling path



from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.networks.layers.factories import Act, Norm
from monai.networks.nets import AutoEncoder
from monai.utils import deprecated_arg



class LinearLatentAE(AutoEncoder):

    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_shape: Sequence[int],
        out_channels: int,
        latent_size: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        inter_channels: Optional[list] = None,
        inter_dilations: Optional[list] = None,
        num_inter_units: int = 2,
        act: Optional[Union[Tuple, str]] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: Optional[Union[Tuple, str, float]] = None,
        bias: bool = True,
        dimensions: Optional[int] = None,
    ) -> None:

        self.in_channels, *self.in_shape = in_shape

        self.latent_size = latent_size
        self.final_size = np.asarray(self.in_shape, dtype=int)
        if dimensions is not None:
            spatial_dims = dimensions

        super().__init__(
            spatial_dims,
            self.in_channels,
            out_channels,
            channels,
            strides,
            kernel_size,
            up_kernel_size,
            num_res_units,
            inter_channels,
            inter_dilations,
            num_inter_units,
            act,
            norm,
            dropout,
            bias,
        )

        padding = same_padding(self.kernel_size)

        for s in strides:
            self.final_size = calculate_out_shape(self.final_size, self.kernel_size, s, padding)  

        linear_size = int(np.product(self.final_size)) * self.encoded_channels

        self.latent_code= nn.Linear(linear_size, self.latent_size)
        self.decodeL = nn.Linear(self.latent_size, linear_size)


    def encode_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encode(x)
        x = self.intermediate(x)
        x = x.view(x.shape[0], -1)
        #mu = self.mu(x)
        #logvar = self.logvar(x)
        #return mu, logvar
        return self.latent_code(x)


    def decode_forward(self, z: torch.Tensor, use_sigmoid: bool = False) -> torch.Tensor:
        x = F.relu(self.decodeL(z))
        x = x.view(x.shape[0], self.channels[-1], *self.final_size)
        x = self.decode(x)
        if use_sigmoid:
            x = torch.sigmoid(x)
        return x


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latentCodes = self.encode_forward(x)
        return self.decode_forward(latentCodes)