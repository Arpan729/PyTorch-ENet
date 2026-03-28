import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

import transforms as ext_transforms
from models.enet import ENet
from train import Train
from test import Test
from metric.iou import IoU
from args import get_arguments
from data.utils import enet_weighing, median_freq_balancing
import utils

# Get the arguments
args = get_arguments()

device = torch.device("cpu")
print(f"Using device: {device}")
#device = torch.device(args.device)


def load_dataset(dataset):
    print("\nLoading dataset...\n")

    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

    label_transform = transforms.Compose([
    transforms.Resize((args.height, args.width), Image.NEAREST),
    ext_transforms.PILToLongTensor(),
    # Extra safety: ensure it's [H, W]
    lambda x: x.squeeze() if x.dim() == 3 else x
])
    # label_transform = transforms.Compose([
    #     transforms.Resize((args.height, args.width), Image.NEAREST),
    #     ext_transforms.PILToLongTensor()
    # ])

    # Get selected dataset
    # Load the training set as tensors
    train_set = dataset(
        args.dataset_dir,
        transform=image_transform,
        label_transform=label_transform)
    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,          # Windows fix
        persistent_workers=False)
    # train_loader = data.DataLoader(
    #     train_set,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.workers)

    # Load the validation set as tensors
    val_set = dataset(
        args.dataset_dir,
        mode='val',
        transform=image_transform,
        label_transform=label_transform)
    val_loader = data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0)

    # val_loader = data.DataLoader(
    #     val_set,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.workers)

    # Load the test set as tensors
    test_set = dataset(
        args.dataset_dir,
        mode='test',
        transform=image_transform,
        label_transform=label_transform)
    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0)

    # test_loader = data.DataLoader(
    #     test_set,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.workers)

    # Get encoding between pixel valus in label images and RGB colors
    class_encoding = train_set.color_encoding

    # Remove the road_marking class from the CamVid dataset as it's merged
    # with the road class
    if args.dataset.lower() == 'camvid':
        del class_encoding['road_marking']

    # Get number of classes to predict
    num_classes = len(class_encoding)

    # Print information for debugging
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    # Get a batch of samples to display
    if args.mode.lower() == 'test':
        images, labels = next(iter(test_loader))  #Edited for comaptibility
        # images, labels = iter(test_loader).next()
    else:
        images, labels = next(iter(train_loader))
    print("Image size:", images.size())
    print("Label size:", labels.size())
    print("Class-color encoding:", class_encoding)


        # Show a batch of samples and labels (Safe version)
    if args.imshow_batch:
        print("Close the figure window to continue...")
        
        # Safe label squeezing - works for both old 3-channel and new single-channel labels
        if labels.dim() == 4:
            if labels.size(1) == 3:           # Old CamVid 3-channel labels
                labels_squeezed = labels[:, 0, :, :]
            else:
                labels_squeezed = labels.squeeze(1)
        else:
            labels_squeezed = labels

        label_to_rgb = transforms.Compose([
            ext_transforms.LongTensorToRGBPIL(class_encoding),
            transforms.ToTensor()
        ])
        
        color_labels = utils.batch_transform(labels_squeezed, label_to_rgb)
        utils.imshow_batch(images, color_labels)
    # Show a batch of samples and labels
    # Show a batch of samples and labels
    # if args.imshow_batch:
    #     print("Close the figure window to continue...")
        
    #     # Fix: squeeze the extra channel dimension from labels
    #     # labels shape: [B, 3, H, W] → [B, H, W]
    #     # labels_squeezed = labels.squeeze(1) if labels.dim() == 4 and labels.size(1) == 3 else labels
    #     if labels.dim() == 4:
    #         if labels.size(1) == 3:                    # Old 3-channel labels
    #             labels_squeezed = labels[:, 0, :, :]   # Take first channel
    #         else:
    #             labels_squeezed = labels.squeeze(1)
    #     else:
    #         labels_squeezed = labels

    #     label_to_rgb = transforms.Compose([
    #         ext_transforms.LongTensorToRGBPIL(class_encoding),
    #         transforms.ToTensor()
    #     ])
    
    # color_labels = utils.batch_transform(labels_squeezed, label_to_rgb)
    # utils.imshow_batch(images, color_labels)
    # # if args.imshow_batch:
    #     print("Close the figure window to continue...")
    #     label_to_rgb = transforms.Compose([
    #         ext_transforms.LongTensorToRGBPIL(class_encoding),
    #         transforms.ToTensor()
    #     ])
    #     color_labels = utils.batch_transform(labels, label_to_rgb)
    #     utils.imshow_batch(images, color_labels)

    # Get class weights from the selected weighing technique
    print("\nWeighing technique:", args.weighing)
    print("Computing class weights...")
    print("(this can take a while depending on the dataset size)")
    class_weights = 0
    if args.weighing.lower() == 'enet':
        class_weights = enet_weighing(train_loader, num_classes)
    elif args.weighing.lower() == 'mfb':
        class_weights = median_freq_balancing(train_loader, num_classes)
    else:
        class_weights = None

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(device)
        # Set the weight of the unlabeled class to 0
        if args.ignore_unlabeled:
            ignore_index = list(class_encoding).index('unlabeled')
            class_weights[ignore_index] = 0

    print("Class weights:", class_weights)

    return (train_loader, val_loader,
            test_loader), class_weights, class_encoding


# def train(train_loader, val_loader, class_weights, class_encoding):
#     print("\nTraining...\n")

#     num_classes = len(class_encoding)

#     # Intialize ENet
#     model = ENet(num_classes).to(device)
#     # Check if the network architecture is correct
#     print(model)

#     # We are going to use the CrossEntropyLoss loss function as it's most
#     # frequentely used in classification problems with multiple classes which
#     # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
#     criterion = nn.CrossEntropyLoss(weight=class_weights)

#     # ENet authors used Adam as the optimizer
#     optimizer = optim.Adam(
#         model.parameters(),
#         lr=args.learning_rate,
#         weight_decay=args.weight_decay)

#     # Learning rate decay scheduler
#     lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs,
#                                      args.lr_decay)

#     # Evaluation metric
#     # if args.ignore_unlabeled:
#     #     ignore_index = list(class_encoding).index('unlabeled')
#     # else:
#     #     ignore_index = 255 if args.ignore_unlabeled else None
#     # metric = IoU(num_classes, ignore_index=ignore_index)

#         # Use 255 as ignore_index since our converted labels use 255 for unlabeled
#     ignore_index = 255
#     criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
#     metric = IoU(num_classes, ignore_index=ignore_index)

#     # Optionally resume from a checkpoint

#         # Optionally resume from a checkpoint
#     if args.resume:
#         try:
#             model, optimizer, start_epoch, best_miou = utils.load_checkpoint(
#                 model, optimizer, args.save_dir, args.name, device)
#             print(f"Resuming from model: Start epoch = {start_epoch} | Best mean IoU = {best_miou:.4f}")
#         except Exception as e:
#             print(f"Warning: Could not load checkpoint ({e}). Starting training from scratch.")
#             start_epoch = 0
#             best_miou = 0
#     else:
#         start_epoch = 0
#         best_miou = 0

#     # if args.resume:
#     #     model, optimizer, start_epoch, best_miou = utils.load_checkpoint(
#     #         model, optimizer, args.save_dir, args.name)
#     #     print("Resuming from model: Start epoch = {0} "
#     #           "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
#     # else:
#     #     start_epoch = 0
#     #     best_miou = 0

#     # Start Training
#     print()
#     train = Train(model, train_loader, optimizer, criterion, metric, device)
#     val = Test(model, val_loader, criterion, metric, device)
#     for epoch in range(start_epoch, args.epochs):
#         print(">>>> [Epoch: {0:d}] Training".format(epoch))

#         epoch_loss, (iou, miou) = train.run_epoch(args.print_step)
#         lr_updater.step()

#         print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
#               format(epoch, epoch_loss, miou))

#         if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
#             print(">>>> [Epoch: {0:d}] Validation".format(epoch))

#             loss, (iou, miou) = val.run_epoch(args.print_step)

#             print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
#                   format(epoch, loss, miou))

#             # Print per class IoU on last epoch or if best iou
#             if epoch + 1 == args.epochs or miou > best_miou:
#                 for key, class_iou in zip(class_encoding.keys(), iou):
#                     print("{0}: {1:.4f}".format(key, class_iou))

#             # Save the model if it's the best thus far
#             if miou > best_miou:
#                 print("\nBest model thus far. Saving...\n")
#                 best_miou = miou
#                 utils.save_checkpoint(model, optimizer, epoch + 1, best_miou,
#                                       args)

#     return model

def train(train_loader, val_loader, class_weights, class_encoding):
    print("\nTraining...\n")

    num_classes = len(class_encoding)

    # Initialize ENet
    model = ENet(num_classes).to(device)

    # === FIXED: Handle class_weights safely ===
    if class_weights is not None:
        if class_weights.shape[0] != num_classes:
            print(f"Warning: class_weights length ({class_weights.shape[0]}) does not match num_classes ({num_classes}). Using uniform weights.")
            class_weights = None
        else:
            class_weights = class_weights.to(device)

    # Create loss
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    # Learning rate scheduler
    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs, args.lr_decay)

    # Evaluation metric - ignore 255 (unlabeled)
    metric = IoU(num_classes, ignore_index=255)

    # Optionally resume from checkpoint
    start_epoch = 0
    best_miou = 0

    if args.resume:
        try:
            model, optimizer, start_epoch, best_miou = utils.load_checkpoint(
                model, optimizer, args.save_dir, args.name, device)
            print(f"Resuming from model: Start epoch = {start_epoch} | Best mean IoU = {best_miou:.4f}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint ({e}). Starting from scratch.")

    # Start Training
    print()
    train_obj = Train(model, train_loader, optimizer, criterion, metric, device)
    val_obj = Test(model, val_loader, criterion, metric, device)

    for epoch in range(start_epoch, args.epochs):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        epoch_loss, (iou, miou) = train_obj.run_epoch(args.print_step)
        lr_updater.step()

        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, epoch_loss, miou))

        # Validation every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))
            loss, (iou, miou) = val_obj.run_epoch(args.print_step)

            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, loss, miou))

            if epoch + 1 == args.epochs or miou > best_miou:
                for key, class_iou in zip(class_encoding.keys(), iou):
                    print("{0}: {1:.4f}".format(key, class_iou))

            # Save best model
            if miou > best_miou:
                print("\nBest model thus far. Saving...\n")
                best_miou = miou
                utils.save_checkpoint(model, optimizer, epoch + 1, best_miou, args)

    return model

def test(model, test_loader, class_weights, class_encoding):
    print("\nTesting...\n")

    num_classes = len(class_encoding)

    # Use ignore_index=255 for unlabeled pixels (very important after conversion)
    ignore_index = 255

    # Create loss - ignore unlabeled class (255)
    if class_weights is not None and class_weights.shape[0] == num_classes:
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    else:
        print(f"Using CrossEntropyLoss with ignore_index=255 (class_weights mismatch)")
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    # Evaluation metric - also ignore 255
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Test the model
    test_obj = Test(model, test_loader, criterion, metric, device)

    print(">>>> Running test dataset")

    loss, (iou, miou) = test_obj.run_epoch(args.print_step)

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

    # Print per class IoU
    for key, class_iou in zip(class_encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))

# def test(model, test_loader, class_weights, class_encoding):
#     print("\nTesting...\n")

#     num_classes = len(class_encoding)

#     # === FIXED: Handle class_weights properly when loading checkpoint ===
#     if class_weights is not None:
#         # Make sure weight tensor has exactly the same number of classes as current model
#         if class_weights.shape[0] != num_classes:
#             print(f"Warning: class_weights length ({class_weights.shape[0]}) "
#                   f"does not match num_classes ({num_classes}). Adjusting...")
#             # Truncate or pad (usually truncate if checkpoint has fewer classes)
#             if class_weights.shape[0] > num_classes:
#                 class_weights = class_weights[:num_classes]
#             else:
#                 # Pad with 1.0 if needed (rare)
#                 padding = torch.ones(num_classes - class_weights.shape[0], 
#                                    device=class_weights.device)
#                 class_weights = torch.cat([class_weights, padding])

#         class_weights = class_weights.to(device)

#     # Create loss
#     criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()

#     # Evaluation metric
#     if args.ignore_unlabeled:
#         ignore_index = list(class_encoding).index('unlabeled')
#     else:
#         ignore_index = None
#     metric = IoU(num_classes, ignore_index=ignore_index)

#     # Test the model
#     test_obj = Test(model, test_loader, criterion, metric, device)

#     print(">>>> Running test dataset")

#     loss, (iou, miou) = test_obj.run_epoch(args.print_step)
#     class_iou = dict(zip(class_encoding.keys(), iou))

#     print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

#     # Print per class IoU
#     for key, class_iou in zip(class_encoding.keys(), iou):
#         print("{0}: {1:.4f}".format(key, class_iou))

# def test(model, test_loader, class_weights, class_encoding):
#     print("\nTesting...\n")

#     num_classes = len(class_encoding)

#     # We are going to use the CrossEntropyLoss loss function as it's most
#     # frequentely used in classification problems with multiple classes which
#     # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
#     criterion = nn.CrossEntropyLoss(weight=class_weights)

#     # Evaluation metric
#     if args.ignore_unlabeled:
#         ignore_index = list(class_encoding).index('unlabeled')
#     else:
#         ignore_index = None
#     metric = IoU(num_classes, ignore_index=ignore_index)

#     # Test the trained model on the test set
#     test = Test(model, test_loader, criterion, metric, device)

#     print(">>>> Running test dataset")

#     loss, (iou, miou) = test.run_epoch(args.print_step)
#     class_iou = dict(zip(class_encoding.keys(), iou))

#     print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

#     # Print per class IoU
#     for key, class_iou in zip(class_encoding.keys(), iou):
#         print("{0}: {1:.4f}".format(key, class_iou))

#     # Show a batch of samples and labels
#     if args.imshow_batch:
#         print("A batch of predictions from the test set...")
#         images, labels = next(iter(test_loader))   # already fixed earlier in some places
#         # images, _ = iter(test_loader).next()
#         predict(model, images, class_encoding)


def predict(model, images, class_encoding):
    images = images.to(device)

    # Make predictions!
    model.eval()
    with torch.no_grad():
        predictions = model(images)

    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(predictions.data, 1)

    label_to_rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    utils.imshow_batch(images.data.cpu(), color_predictions)


# Run only if this module is being run directly
if __name__ == '__main__':

    # Fail fast if the dataset directory doesn't exist
    assert os.path.isdir(
        args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(
            args.dataset_dir)

    # Fail fast if the saving directory doesn't exist
    assert os.path.isdir(
        args.save_dir), "The directory \"{0}\" doesn't exist.".format(
            args.save_dir)

    # Import the requested dataset
    if args.dataset.lower() == 'camvid':
        from data import CamVid as dataset
    elif args.dataset.lower() == 'cityscapes':
        from data import Cityscapes as dataset
    else:
        # Should never happen...but just in case it does
        raise RuntimeError("\"{0}\" is not a supported dataset.".format(
            args.dataset))

    loaders, w_class, class_encoding = load_dataset(dataset)
    train_loader, val_loader, test_loader = loaders

    if args.mode.lower() in {'train', 'full'}:
        model = train(train_loader, val_loader, w_class, class_encoding)

    if args.mode.lower() in {'test', 'full'}:
        if args.mode.lower() == 'test':
            # Intialize a new ENet model
            num_classes = len(class_encoding)
            model = ENet(num_classes).to(device)

        # Initialize a optimizer just so we can retrieve the model from the
        # checkpoint
        optimizer = optim.Adam(model.parameters())

        # Load the previoulsy saved model state to the ENet model
        model = utils.load_checkpoint(model, optimizer, args.save_dir, args.name, device)[0]
        # # model = utils.load_checkpoint(model, optimizer, args.save_dir,
        #                               args.name)[0]

        if args.mode.lower() == 'test':
            print(model)

        test(model, test_loader, w_class, class_encoding)
