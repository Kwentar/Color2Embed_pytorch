import argparse
import datetime
import os
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from datasets import Color2EmbedDataset
from losses import Color2EmbedLoss
from models import Color2Embed
from utils import AverageMeter


def print_verbose(text, verbose=False):
    if verbose:
        print(text)


def create_sampler_and_dataloader(dataset, batch_size_per_gpu, num_workers=4, shuffle=True):
    train_sampler = DistributedSampler(dataset,
                                       shuffle=shuffle,
                                       drop_last=True, seed=42)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size_per_gpu,
                            num_workers=num_workers,
                            pin_memory=True,
                            sampler=train_sampler,
                            drop_last=True)

    return dataloader


def train(args):
    world_size = int(os.environ['WORLD_SIZE'])
    rank = args.local_rank
    global_rank = int(os.environ['RANK'])
    master_ip = os.environ["MASTER_ADDR"]
    verbose = global_rank == 0
    print_verbose('Started', verbose)

    experiment_folder = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    log_dir = os.path.join('logs/', experiment_folder)
    writer = None
    if verbose:
        os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        latest_dir = "logs/latest"
        if os.path.exists(latest_dir):
            os.unlink(latest_dir)
        os.symlink(os.path.join(os.getcwd(), log_dir), latest_dir, target_is_directory=True)

    torch.cuda.set_device(rank)

    scaler = torch.cuda.amp.GradScaler()
    print_verbose('Start init ddp', verbose)
    dist.init_process_group(backend='nccl',
                            init_method=f'tcp://{master_ip}:23461',
                            rank=global_rank,
                            world_size=world_size)
    print_verbose('finish init ddp', verbose)
    dataset = Color2EmbedDataset(verbose)
    dataloader = create_sampler_and_dataloader(dataset, config.TRAIN_BATCH_SIZE_PER_GPU)

    model = Color2Embed(color_embedding_dim=config.COLOR_EMBEDDING_DIM)
    loss = Color2EmbedLoss(rank)
    loss.to(rank)
    model.to(rank)

    ddp_model = DistributedDataParallel(model, broadcast_buffers=False, device_ids=[rank])

    optimizer = Adam(ddp_model.parameters(),
                     lr=config.TRAIN_LR)

    total_iter = len(dataset) // config.TRAIN_BATCH_SIZE_PER_GPU // world_size * config.TRAIN_EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=total_iter)

    train_start = time.time()
    train_start_time = datetime.datetime.now()
    print_verbose(f'Train started at {train_start_time}', verbose)
    global_step = 0

    avg_loss = AverageMeter()
    # images_step = 0
    for epoch in range(config.TRAIN_EPOCHS):
        draw_images = True
        print_verbose(f'Start epoch {epoch + 1}', verbose)
        dataloader.sampler.set_epoch(epoch)

        tq = tqdm(dataloader, disable=not verbose)

        for item in tq:
            global_step += 1

            l_channel, ground_truth_ab, ground_truth_rgb, color_source = item

            # print(l_channel.shape, ground_truth_ab.shape, ground_truth_rgb.shape, color_source.shape)

            #with torch.cuda.amp.autocast():
            pab = ddp_model(l_channel.to(rank), color_source.to(rank))

            merged_lab = torch.cat((l_channel.to(rank), pab), 1)
            prgb = merged_lab.detach().cpu().numpy()
            prgb = np.transpose(prgb, (0, 2, 3, 1))
            for i in range(prgb.shape[0]):
                prgb[i] = cv2.cvtColor(np.clip(prgb[i]*255, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)

                merged_lab[i] = dataset.transform_torch(image=prgb[i])['image'].to(rank)/255.
            # merged_lab = torch.from_numpy(prgb).to(rank)
            loss_value, l_per, l_rec = loss(pab, ground_truth_ab.to(rank), merged_lab.to(rank), ground_truth_rgb.to(rank))
            loss_value.backward()
            # scaler.scale(loss_value).backward()

            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            loss_v = loss_value.detach().cpu()
            avg_loss.update(loss_v)
            tq.set_postfix_str(f'loss: {loss_v:.3f}')
            if verbose:
                writer.add_scalar('train/loss_step', loss_v, global_step)
                writer.add_scalar('pab/pab_min', pab.detach().cpu().min(), global_step)
                writer.add_scalar('pab/pab_max', pab.detach().cpu().max(), global_step)
                writer.add_scalar('train/l_per', l_per.detach().cpu().max(), global_step)
                writer.add_scalar('train/l_rec', l_rec.detach().cpu().max(), global_step)

            if global_step % 2000 == 1 and verbose:
                cv2.imwrite('test.png', prgb[0])
                pa_channel = pab[:, 0, :, :].unsqueeze(1)
                pb_channel = pab[:, 1, :, :].unsqueeze(1)
                grid = torchvision.utils.make_grid(merged_lab[:, [2, 1, 0]], normalize=True, scale_each=True)
                writer.add_image("images_prgb", grid, global_step=epoch)
                grid = torchvision.utils.make_grid(ground_truth_rgb[:, [2, 1, 0]], normalize=True, scale_each=True)
                writer.add_image("images_ground_truth_rgb", grid, global_step=epoch)
                grid = torchvision.utils.make_grid(color_source[:, [2, 1, 0]], normalize=True, scale_each=True)
                writer.add_image("images_color_source", grid, global_step=epoch)
                grid = torchvision.utils.make_grid(l_channel.expand(-1, 3, -1, -1)[:, [2, 1, 0]], normalize=True, scale_each=True)
                writer.add_image("images_l_channel", grid, global_step=epoch)
                grid = torchvision.utils.make_grid(pa_channel.expand(-1, 3, -1, -1)[:, [2, 1, 0]], normalize=True, scale_each=True)
                writer.add_image("images_a_channel", grid, global_step=epoch)
                grid = torchvision.utils.make_grid(pb_channel.expand(-1, 3, -1, -1)[:, [2, 1, 0]], normalize=True, scale_each=True)
                writer.add_image("images_b_channel", grid, global_step=epoch)
                draw_images = False

            if global_step % 50 == 0 and verbose:
                writer.add_scalar('train/loss', avg_loss.avg, global_step)
                writer.add_scalar('train/lr', get_lr(optimizer), global_step)
                writer.add_scalar('train/epoch', epoch, global_step)
                avg_loss.reset()
                writer.flush()

        if verbose:
            delta_time = time.time() - train_start
            finish_time = train_start_time + config.TRAIN_EPOCHS * datetime.timedelta(
                seconds=int(delta_time / (epoch + 1)))
            print(f'Estimated finish time: {finish_time}')

            checkpoints_dir = os.path.join(log_dir, 'models')
            os.makedirs(checkpoints_dir, exist_ok=True)
            torch.save(ddp_model.module.state_dict(), os.path.join(checkpoints_dir, f'weights_{epoch}.pth'))

    print_verbose(f'Train finished at {datetime.datetime.now()}, started at {train_start_time}, '
                  f'total {(datetime.datetime.now() - train_start_time).seconds / 3600:.2f} hours', verbose)
    dist.destroy_process_group()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args_ = parser.parse_args()
    train(args_)
