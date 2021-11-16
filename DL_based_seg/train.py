import numpy as np
import logging
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from utils import AverageMeter
from utils import compute_params, validation
from model import get_model
from data import create_loaders
from get_args import get_arguments

def train_model(
    net,
    train_loader,
    optim,
    epoch,
    segm_crit,
):

    # get the epoch info
    logger = logging.getLogger(__name__)
    logger.info(
        "Epoch{}".format(
            epoch
        )
    )

    # define the losses and tbar to show
    losses = AverageMeter()
    tbar = tqdm(train_loader)

    # begin
    net.train()
    for i, sample in enumerate(tbar):
        image = sample["image"].float().cuda()
        target = sample["label"].cuda()
        target_var = torch.autograd.Variable(target).long()
        # Compute output
        output = net(image)
        soft_output = nn.LogSoftmax()(output)

        # Compute loss and backpropagate
        loss = segm_crit(soft_output, target_var)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # update the losses and tbar
        losses.update(loss.item())
        tbar.set_description('Train loss: %.3f' % (losses.avg))

def main():

    # get the args and the logger
    args = get_arguments()
    logger = logging.getLogger(__name__)
    logger.debug(args)

    # Set-up random seeds
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # create the dataloader and the model
    train_loader, val_loader, num_class = create_loaders(args)
    net = get_model()

    # compute the model size(param number)
    logger.info(
        " Loaded Encoder with #TOTAL PARAMS={:3.2f}M".format(
            compute_params(net)[0] / 1e6
        )
    )

    #define the loss and the optimzator
    segm_crit = nn.NLLLoss(ignore_index=255).cuda()
    optim = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    best_miou = 0

    # train 400 epoches
    for epoch in range(400):

        # train
        train_model(
            net,
            train_loader,
            optim,
            epoch,
            segm_crit,
        )

        # validate and save
        if (epoch + 1) % (1) == 0:
            logger.info(
                " Validating epoch{}".format(
                    str(epoch)
                )
            )
            task_miou = validation(
                args,
                net,
                val_loader,
                epoch,
                segm_crit
            )
            if task_miou > best_miou:
                PATH = args.ckpt_path
                torch.save(net.state_dict(),PATH)
                best_miou = task_miou
            logger.info(
                " current best val miou={}".format(
                    task_miou
                )
            )


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s",
        level=logging.INFO,
    )
    main()
