# import torch
# import numpy as np
# from metric import metric
# from metric.confusionmatrix import ConfusionMatrix


# class IoU(metric.Metric):
#     """Computes the intersection over union (IoU) per class and corresponding
#     mean (mIoU).

#     Intersection over union (IoU) is a common evaluation metric for semantic
#     segmentation. The predictions are first accumulated in a confusion matrix
#     and the IoU is computed from it as follows:

#         IoU = true_positive / (true_positive + false_positive + false_negative).

#     Keyword arguments:
#     - num_classes (int): number of classes in the classification problem
#     - normalized (boolean, optional): Determines whether or not the confusion
#     matrix is normalized or not. Default: False.
#     - ignore_index (int or iterable, optional): Index of the classes to ignore
#     when computing the IoU. Can be an int, or any iterable of ints.
#     """

#     def __init__(self, num_classes, normalized=False, ignore_index=None):
#         super().__init__()
#         self.conf_metric = ConfusionMatrix(num_classes, normalized)

#         if ignore_index is None:
#             self.ignore_index = None
#         elif isinstance(ignore_index, int):
#             self.ignore_index = (ignore_index,)
#         else:
#             try:
#                 self.ignore_index = tuple(ignore_index)
#             except TypeError:
#                 raise ValueError("'ignore_index' must be an int or iterable")

#     def reset(self):
#         self.conf_metric.reset()

#     def add(self, predicted, target):
#         """Adds the predicted and target pair to the IoU metric.

#         Keyword arguments:
#         - predicted (Tensor): Can be a (N, K, H, W) tensor of
#         predicted scores obtained from the model for N examples and K classes,
#         or (N, H, W) tensor of integer values between 0 and K-1.
#         - target (Tensor): Can be a (N, K, H, W) tensor of
#         target scores for N examples and K classes, or (N, H, W) tensor of
#         integer values between 0 and K-1.

#         """
#         # Dimensions check
#         assert predicted.size(0) == target.size(0), \
#             'number of targets and predicted outputs do not match'
#         assert predicted.dim() == 3 or predicted.dim() == 4, \
#             "predictions must be of dimension (N, H, W) or (N, K, H, W)"
#         assert target.dim() == 3 or target.dim() == 4, \
#             "targets must be of dimension (N, H, W) or (N, K, H, W)"

#         # If the tensor is in categorical format convert it to integer format
#         if predicted.dim() == 4:
#             _, predicted = predicted.max(1)
#         if target.dim() == 4:
#             _, target = target.max(1)

#         self.conf_metric.add(predicted.view(-1), target.view(-1))

#     def value(self):
#         """Computes the IoU and mean IoU.

#         The mean computation ignores NaN elements of the IoU array.

#         Returns:
#             Tuple: (IoU, mIoU). The first output is the per class IoU,
#             for K classes it's numpy.ndarray with K elements. The second output,
#             is the mean IoU.
#         """
#         conf_matrix = self.conf_metric.value()
#         if self.ignore_index is not None:
#             conf_matrix[:, self.ignore_index] = 0
#             conf_matrix[self.ignore_index, :] = 0
#         true_positive = np.diag(conf_matrix)
#         false_positive = np.sum(conf_matrix, 0) - true_positive
#         false_negative = np.sum(conf_matrix, 1) - true_positive

#         # Just in case we get a division by 0, ignore/hide the error
#         with np.errstate(divide='ignore', invalid='ignore'):
#             iou = true_positive / (true_positive + false_positive + false_negative)

#         return iou, np.nanmean(iou)



# import torch
# from .confusionmatrix import ConfusionMatrix


# class IoU:
#     """Computes the intersection over union (IoU) per class and mean IoU."""

#     def __init__(self, num_classes, ignore_index=None):
#         self.num_classes = num_classes
#         self.ignore_index = ignore_index
#         self.conf_metric = ConfusionMatrix(num_classes)

#     def reset(self):
#         self.conf_metric.reset()

#     def add(self, predicted, target):
#         """Adds a batch of predictions and targets to the metric."""
#         # Remove ignored pixels before adding to confusion matrix
#         if self.ignore_index is not None:
#             mask = (target != self.ignore_index)
#             predicted = predicted[mask]
#             target = target[mask]

#         # Now target should only contain values 0 to num_classes-1
#         self.conf_metric.add(predicted.view(-1), target.view(-1))

#     def value(self):
#         """Returns the IoU for each class and the mean IoU."""
#         conf_matrix = self.conf_metric.value()
#         # Compute IoU
#         true_positive = torch.diag(conf_matrix)
#         false_positive = conf_matrix.sum(dim=1) - true_positive
#         false_negative = conf_matrix.sum(dim=0) - true_positive

#         iou = true_positive / (true_positive + false_positive + false_negative + 1e-10)
#         miou = iou.mean()

#         return iou.cpu().numpy(), miou.cpu().numpy()


import torch
from .confusionmatrix import ConfusionMatrix


class IoU:
    """Computes the intersection over union (IoU) per class and mean IoU."""

    def __init__(self, num_classes, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.conf_metric = ConfusionMatrix(num_classes)

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """predicted: [B, C, H, W], target: [B, H, W]"""
        # Convert logits to class predictions
        _, predicted = torch.max(predicted, 1)   # shape becomes [B, H, W]

        # Ignore unlabeled pixels (255)
        if self.ignore_index is not None:
            mask = (target != self.ignore_index)
            predicted = predicted[mask]
            target = target[mask]

        self.conf_metric.add(predicted.view(-1), target.view(-1))

    def value(self):
        """Returns per-class IoU and mean IoU."""
        conf_matrix = self.conf_metric.value()          # This returns numpy array

        # Convert to tensor for torch operations
        conf_matrix = torch.from_numpy(conf_matrix).float()

        true_positive = torch.diag(conf_matrix)
        false_positive = conf_matrix.sum(dim=1) - true_positive
        false_negative = conf_matrix.sum(dim=0) - true_positive

        # IoU per class
        iou = true_positive / (true_positive + false_positive + false_negative + 1e-10)
        
        # Mean IoU
        miou = iou.mean().item()

        return iou.cpu().numpy(), miou