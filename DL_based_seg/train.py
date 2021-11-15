from data import create_loaders
from utils import AverageMeter
from utils import compute_params
from model import get_model
from tqdm import tqdm

import argparse
import logging
import torch
import torch.nn as nn
import time
import numpy as np
import random

n_class = 2
def train_model(
    net,
    train_loader,
    optim,
    epoch,
    segm_crit,
    print_every=10,
):

    try:
        train_loader.dataset.set_stage("train")
    except AttributeError:
        train_loader.dataset.dataset.set_stage("train")  # for subset
    net.train()
    # freeze_bn = True
    # if freeze_bn:
    #     for m in mobilenet.modules():
    #         if isinstance(m, nn.BatchNorm2d):
    #             m.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    tbar = tqdm(train_loader)

    for i, sample in enumerate(tbar):
        start = time.time()
        image = sample["image"].float().cuda()
        target = sample["mask"].cuda()
        target_var = torch.autograd.Variable(target).float()
        # Compute output
        output = net(image)

        if isinstance(output, tuple):
            output, aux_outs = output

        target_var = nn.functional.interpolate(
            target_var[:, None], size=output.size()[2:], mode="nearest"
        ).long()[:, 0]

        soft_output = nn.LogSoftmax()(output)
        # Compute loss and backpropagate
        loss = segm_crit(soft_output, target_var)
        optim.zero_grad()

        loss.backward()

        optim.step()

        losses.update(loss.item())

        tbar.set_description('Train loss: %.3f' % (losses.avg))
        batch_time.update(time.time() - start)

        logger = logging.getLogger(__name__)

        if i % print_every == 0:
            logger.info(
                " Train epoch: {} [{}/{}]\t"
                "Avg. Loss: {:.3f}\t"
                "Avg. Time: {:.3f}".format(
                    epoch, i, len(train_loader), losses.avg, batch_time.avg
                )
            )


def main():

    args = get_arguments()
    logger = logging.getLogger(__name__)
    logger.debug(args)

    # Set-up random seeds
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # net = mobilenet_v2(False)

    net = get_model()


    logger.info(
        " Loaded Encoder with #TOTAL PARAMS={:3.2f}M".format(
            compute_params(net)[0] / 1e6
        )
    )

    train_loader, val_loader = create_loaders(args)
    segm_crit = nn.NLLLoss2d(ignore_index=255).cuda()


    best_miou = 0

    for epoch in range(100):
        optim = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
        train_loader.batch_sampler.batch_size = args.batch_size[1]

        train_model(
            net,
            train_loader,
            optim,
            epoch,
            segm_crit,
            print_every= 200,
        )
        if (epoch + 1) % (1) == 0:
            logger.info(
                " Validating mobilenet_v2 epoch{}".format(
                    str(epoch)
                )
            )
            task_miou = validate(
                net,
                val_loader,
                1,
                epoch,
                num_classes=2,
                print_every=100,
                omit_classes=[0],
            )
            if task_miou > best_miou:
                PATH = "../best_hrnet_ach.pth"
                torch.save(net.state_dict(),PATH)
                logger.info(
                    " current best val miou={}".format(
                        task_miou
                    )
                )
                best_miou = task_miou

if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s",
        level=logging.INFO,
    )
    main()
