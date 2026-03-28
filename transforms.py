import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
from torchvision.transforms import ToPILImage


# class PILToLongTensor(object):
#     """Converts a ``PIL Image`` to a ``torch.LongTensor``.

#     Code adapted from: http://pytorch.org/docs/master/torchvision/transforms.html?highlight=totensor

#     """

#     def __call__(self, pic):
#         """Performs the conversion from a ``PIL Image`` to a ``torch.LongTensor``.

#         Keyword arguments:
#         - pic (``PIL.Image``): the image to convert to ``torch.LongTensor``

#         Returns:
#         A ``torch.LongTensor``.

#         """
#         if not isinstance(pic, Image.Image):
#             raise TypeError("pic should be PIL Image. Got {}".format(
#                 type(pic)))

#         # handle numpy array
#         if isinstance(pic, np.ndarray):
#             img = torch.from_numpy(pic.transpose((2, 0, 1)))
#             # backward compatibility
#             return img.long()

#         # Convert PIL image to ByteTensor
#         img = torch.frombuffer(pic.tobytes(), dtype=torch.uint8).clone() #Edited for compatibility
#         # img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

#         # Reshape tensor
#         nchannel = len(pic.mode)
#         img = img.view(pic.size[1], pic.size[0], nchannel)

#         # Convert to long and squeeze the channels
#         return img.transpose(0, 1).transpose(0,
#                                              2).contiguous().long().squeeze_()

class PILToLongTensor(object):
    """Converts a ``PIL Image`` to a ``torch.LongTensor`` of shape [H, W]."""

    def __call__(self, pic):
        if not isinstance(pic, Image.Image):
            raise TypeError("pic should be PIL Image. Got {}".format(type(pic)))

        # handle numpy array
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.long().squeeze(0)   # ensure single channel

        # Convert PIL image to tensor
        img = torch.frombuffer(pic.tobytes(), dtype=torch.uint8).clone()

        # Reshape
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        # Transpose to [C, H, W] then squeeze to [H, W]
        img = img.transpose(0, 1).transpose(0, 2).contiguous().long()

        # Force single channel - this fixes the CamVid 3-channel problem
        if img.dim() == 3 and img.size(0) == 3:
            img = img[0]          # All 3 channels are identical in label images
        elif img.dim() == 3:
            img = img.squeeze(0)

        return img
    
    
class LongTensorToRGBPIL(object):
    """Converts a ``torch.LongTensor`` to a ``PIL image``.

    The input is a ``torch.LongTensor`` where each pixel's value identifies the
    class.

    Keyword arguments:
    - rgb_encoding (``OrderedDict``): An ``OrderedDict`` that relates pixel
    values, class names, and class colors.

    """
    def __init__(self, rgb_encoding):
        self.rgb_encoding = rgb_encoding

    def __call__(self, tensor):
    
    # Check if label_tensor is a LongTensor
        if not isinstance(tensor, torch.LongTensor):
            # In case it's a FloatTensor or something else (rare)
            tensor = tensor.long()

        if not isinstance(self.rgb_encoding, OrderedDict):
            raise TypeError("encoding should be an OrderedDict. Got {}".format(
                type(self.rgb_encoding)))

        # === Robust shape handling ===
        # Possible incoming shapes from batch_transform + unbind:
        #   [H, W]      -> ideal
        #   [1, H, W]   -> common
        #   [3, H, W]   -> happens with CamVid labels after Resize
        #   [B, H, W] or others -> shouldn't reach here

        orig_shape = tensor.shape

        if tensor.dim() == 4:                     # [B, C, H, W] — shouldn't happen
            tensor = tensor[0]                    # take first sample
        if tensor.dim() == 3:
            if tensor.size(0) == 3:               # [3, H, W] — CamVid case
                tensor = tensor[0]                # take first channel (all channels are identical)
            elif tensor.size(0) == 1:             # [1, H, W]
                tensor = tensor.squeeze(0)
            # else: assume it's already [C, H, W] with C != 3, but unlikely

        # Now tensor should be 2D: [H, W]
        if tensor.dim() != 2:
            raise ValueError(f"Unexpected tensor shape after processing: {orig_shape}")

        H, W = tensor.shape
        color_tensor = torch.zeros(3, H, W, dtype=torch.uint8)   # Better: use zeros instead of ByteTensor

        for index, (class_name, color) in enumerate(self.rgb_encoding.items()):
            # mask shape will be [H, W]
            mask = (tensor == index)

            for channel, color_value in enumerate(color):
                color_tensor[channel].masked_fill_(mask, color_value)

        return ToPILImage()(color_tensor)

    # def __call__(self, tensor):
    #     """Performs the conversion from ``torch.LongTensor`` to a ``PIL image``

    #     Keyword arguments:
    #     - tensor (``torch.LongTensor``): the tensor to convert

    #     Returns:
    #     A ``PIL.Image``.

    #     """
    #     # Check if label_tensor is a LongTensor
    #     if not isinstance(tensor, torch.LongTensor):
    #         raise TypeError("label_tensor should be torch.LongTensor. Got {}"
    #                         .format(type(tensor)))
    #     # Check if encoding is a ordered dictionary
    #     if not isinstance(self.rgb_encoding, OrderedDict):
    #         raise TypeError("encoding should be an OrderedDict. Got {}".format(
    #             type(self.rgb_encoding)))

    #     # label_tensor might be an image without a channel dimension, in this
    #     # case unsqueeze it
    #     if len(tensor.size()) == 2:
    #         tensor.unsqueeze_(0)

    #     color_tensor = torch.ByteTensor(3, tensor.size(1), tensor.size(2))

    #     for index, (class_name, color) in enumerate(self.rgb_encoding.items()):
    #         # Get a mask of elements equal to index
    #         mask = torch.eq(tensor, index).squeeze_()
    #         # Fill color_tensor with corresponding colors
    #         for channel, color_value in enumerate(color):
    #             color_tensor[channel][mask] = color_value
    #             #color_tensor[channel].masked_fill_(mask, color_value)

    #     return ToPILImage()(color_tensor)
