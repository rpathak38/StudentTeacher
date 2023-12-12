import sys
import os
import time
import numpy as np
import glob

import torch
import torch.nn as nn

from Data import dataloaders
from Models import models
from Metrics import performance_metrics
from Metrics import losses
from Student_Models import unet, unet_attention, unet_inception, unet_skip
import types


def train_epoch(model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss):
    t = time.time()
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = Dice_loss(output, target) + BCE_loss(torch.sigmoid(output), target)
        loss.backward()
        optimizer.step()
        loss_accumulator.append(loss.item())
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(loss_accumulator)


def student_train_epoch(student_model, teacher_model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss, KLT_loss, Temperature, alpha):
    t = time.time()
    student_model.train()
    teacher_model.eval()
    loss_accumulator = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_output = teacher_model(data)
        student_output = student_model(data)
        
        ground_truth_loss = Dice_loss(student_output, target) + BCE_loss(torch.sigmoid(student_output), target)
        teach_loss = KLT_loss(student_output, teacher_output)
        loss = ground_truth_loss * (1 - alpha) + teach_loss * alpha
        loss.backward() 
        optimizer.step()
        loss_accumulator.append(loss.item())
        
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(loss_accumulator)


@torch.no_grad()
def test(model, device, test_loader, epoch, perf_measure):
    t = time.time()
    model.eval()
    perf_accumulator = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        perf_accumulator.append(perf_measure(output, target).item())
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(perf_accumulator), np.std(perf_accumulator)


def xavier_init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)

def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.dataset == "Kvasir":
        kvasir_path = args.root + "Kvasir-SEG/"
        img_path = kvasir_path + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = kvasir_path + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "CVC":
        cvc_path = args.root + "CVC-ClinicDB/"
        img_path = cvc_path + "Original/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = cvc_path + "Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "both":
        input_paths = []
        target_paths = []

        kvasir_path = args.root + "Kvasir-SEG/"
        img_path = kvasir_path + "images/*"
        input_paths += sorted(glob.glob(img_path))
        depth_path = kvasir_path + "masks/*"
        target_paths += sorted(glob.glob(depth_path))

        cvc_path = args.root + "CVC-ClinicDB/"
        img_path = cvc_path + "Original/*"
        input_paths += sorted(glob.glob(img_path))
        depth_path = cvc_path + "Ground Truth/*"
        target_paths += sorted(glob.glob(depth_path))

    train_dataloader, test_dataloader, val_dataloader = dataloaders.get_dataloaders(
        input_paths, target_paths, batch_size=args.batch_size
    )

    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()
    KLT_loss = losses.KLTDivergence(temperature=args.temp)

    perf = performance_metrics.DiceScore()

    model = None
    teach_model = None
    if args.model == "teacher":
        model = models.FCBFormer()
    else:
        teach_model = models.FCBFormer()
        saved_state = torch.load("Trained models/FCBFormer_" + args.dataset + ".pt")
        # Update model state
        teach_model.load_state_dict(saved_state["model_state_dict"])

        if args.model == "unet":
            model = unet.UNet(3, 1, [64, 128, 256])
        elif args.model == "unet_attention":
            model = unet_attention.UNetAttn(3, 1, [64, 128, 256])
        elif args.model == "unet_inception":
            model = unet_inception.UNetInception(3, 1, [64, 128, 256])
        elif args.model == "unet_skip":
            model = unet_skip.UNetSkip(3, 1, [64, 128, 256])
        xavier_init_weights(model)

    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)
    if teach_model is not None:
        teach_model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # student case
    if teach_model is not None:
        return (
            device,
            train_dataloader,
            test_dataloader,
            val_dataloader,
            Dice_loss,
            BCE_loss,
            KLT_loss,
            perf,
            model,
            teach_model,
            optimizer,
            args.alpha,
            args.temp
        )
    else:
        return (
            device,
            train_dataloader,
            test_dataloader,
            val_dataloader,
            Dice_loss,
            BCE_loss,
            perf,
            model,
            optimizer,
        )


# def train(args):
#     (
#         device,
#         train_dataloader,
#         val_dataloader,
#         Dice_loss,
#         BCE_loss,
#         perf,
#         model,
#         optimizer,
#     ) = build(args)
#
#     if not os.path.exists("./Trained models"):
#         os.makedirs("./Trained models")
#
#     prev_best_test = None
#     if args.lrs == "true":
#         if args.lrs_min > 0:
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer, mode="max", factor=0.5, min_lr=args.lrs_min, verbose=True
#             )
#         else:
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer, mode="max", factor=0.5, verbose=True
#             )
#     for epoch in range(1, args.epochs + 1):
#         try:
#             loss = train_epoch(
#                 model, device, train_dataloader, optimizer, epoch, Dice_loss, BCE_loss
#             )
#             test_measure_mean, test_measure_std = test(
#                 model, device, val_dataloader, epoch, perf
#             )
#         except KeyboardInterrupt:
#             print("Training interrupted by user")
#             sys.exit(0)
#         if args.lrs == "true":
#             scheduler.step(test_measure_mean)
#         if prev_best_test == None or test_measure_mean > prev_best_test:
#             print("Saving...")
#             torch.save(
#                 {
#                     "epoch": epoch,
#                     "model_state_dict": model.state_dict()
#                     if args.mgpu == "false"
#                     else model.module.state_dict(),
#                     "optimizer_state_dict": optimizer.state_dict(),
#                     "loss": loss,
#                     "test_measure_mean": test_measure_mean,
#                     "test_measure_std": test_measure_std,
#                 },
#                 "Trained models/FCBFormer_" + args.dataset + ".pt",
#             )
#             prev_best_test = test_measure_mean


def setup_train_args(my_model="teacher", temperature=1, alpha=0.5, dataset="Kvasir", data_root="./data_root/Kvasir-SEG/", epochs=200,
                     batch_size=16,
                     learning_rate=1e-4, learning_rate_scheduler=True, learning_rate_scheduler_minimum=1e-6,
                     multi_gpu=False):
    args = dict()
    args["dataset"] = dataset
    args["root"] = data_root
    args["epochs"] = epochs
    args["batch_size"] = batch_size
    args["lr"] = learning_rate
    args['lrs'] = learning_rate_scheduler
    args['lrs_min'] = learning_rate_scheduler_minimum
    args['mgpu'] = multi_gpu
    args['model'] = my_model
    args['temp'] = temperature ## controls how much the sigmoid is softened, higher temperature means softer sigmoid
    args['alpha'] = alpha ## controls we weight the teacher's outputs as opposed to ground truth, higher alpha means higher emphasis on teacher's output (must be between 0-1)
    return types.SimpleNamespace(**args)

# def get_args():
#     parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
#     parser.add_argument("--dataset", type=str, required=True, choices=["Kvasir", "CVC"])
#     parser.add_argument("--data-root", type=str, required=True, dest="root")
#     parser.add_argument("--epochs", type=int, default=200)
#     parser.add_argument("--batch-size", type=int, default=16)
#     parser.add_argument("--learning-rate", type=float, default=1e-4, dest="lr")
#     parser.add_argument(
#         "--learning-rate-scheduler", type=str, default="true", dest="lrs"
#     )
#     parser.add_argument(
#         "--learning-rate-scheduler-minimum", type=float, default=1e-6, dest="lrs_min"
#     )
#     parser.add_argument(
#         "--multi-gpu", type=str, default="false", dest="mgpu", choices=["true", "false"]
#     )
#
#     return parser.parse_args()


# def main():
#     args = get_args()
#     train(args)
#
#
# if __name__ == "__main__":
#     main()
