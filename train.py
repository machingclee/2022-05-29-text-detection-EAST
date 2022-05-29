import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import CustomDataset
from detect import performance_check
from models import EAST
from losses import Loss
from tqdm import tqdm
from device import device

import os
import time

from utils import ConsoleLog

console_log = ConsoleLog(lines_up_on_end=1)


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval):
    file_num = len(os.listdir(train_img_path))
    trainset = CustomDataset(train_img_path, train_gt_path)
    train_loader = data.DataLoader(trainset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   drop_last=True)

    criterion = Loss()
    model = EAST()
    data_parallel = False

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[epoch_iter // 2],
                                         gamma=0.1)

    for epoch in range(epoch_iter):
        model.train()
        epoch_loss = 0
        epoch_time = time.time()

        for batch, (img, gt_score, gt_geo, ignored_map) in enumerate(tqdm(
            train_loader,
            total=len(trainset) // batch_size,
            bar_format="{desc}: {percentage:.1f}%|{bar:15}| {n}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]"
        )):
            start_time = time.time()

            img = img.to(device)
            gt_score = gt_score.to(device)
            gt_geo = gt_geo.to(device)
            ignored_map = ignored_map.to(device)

            pred_score, pred_geo = model(img)

            loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (batch + 1) % save_interval == 0:
                performance_check(model, save_image_path="results/epoch_{}_batch_{}.jpg".format(epoch, batch + 1))

            console_log.print(
                'Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(
                    epoch + 1, epoch_iter, batch + 1, int(file_num / batch_size), time.time() - start_time, loss.item()),
                is_key_value=False
            )

        if (epoch + 1) % interval == 0:
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch + 1)))


if __name__ == '__main__':
    train_img_path = os.path.abspath('dataset/images')
    train_gt_path = os.path.abspath('dataset/annotations')
    pths_path = './pths'
    batch_size = 24
    lr = 1e-3
    num_workers = 4
    epoch_iter = 600
    save_interval = 5
    train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval)
