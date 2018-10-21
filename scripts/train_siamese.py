#!/usr/bin/env python3
import argparse
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.extend(['.', '..'])
import lowshot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, action='store', default=32)
    parser.add_argument('--k_shots', type=int, action='store', default=1)
    parser.add_argument('--n_ways', type=int, action='store', default=5)
    parser.add_argument('--workers', type=int, action='store', default=4)
    parser.add_argument('--epochs', type=int, action='store', default=20)
    parser.add_argument('--omniglot', type=str, action='store', required=True)
    parser.add_argument('--type', type=str, action='store', default='weighted_l1')
    parser.add_argument('--save_path', type=str, action='store', required=True)
    parser.add_argument('--margin', type=float, action='store', default=1.0)
    parser.add_argument('--early_stopping', type=int, action='store', default=4)

    return parser.parse_args()


def train(args):
    device = lowshot.utils.get_device()
    print(f'Device: {device}.')

    train_set, test_set, mapping = lowshot.utils.parse_omniglot(args.omniglot)
    train_set, val_set = lowshot.utils.extract_validation(*train_set)

    loader = DataLoader(lowshot.siamese.SiameseDataset(*train_set),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )
    val_tester = lowshot.siamese.SiameseTest(
        *val_set,
        n_ways=args.n_ways,
        k_shots=args.k_shots,
        device=device
    )
    test_tester = lowshot.siamese.SiameseTest(
        *test_set,
        n_ways=args.n_ways,
        k_shots=args.k_shots,
        device=device
    )

    net = lowshot.siamese.OmniglotSiamese(type=args.type).to(device)
    optimizer = torch.optim.Adam(net.parameters())

    best_acc = 0.
    no_improv_counter = 0
    for epoch in range(args.epochs):
        net.train()

        print(f'Epoch {epoch}:', end=' ')
        epoch_loss = 0.

        for data in loader:
            img_1, img_2, dist = data['img_1'].to(device), data['img_2'].to(device), data['dist'].to(device)

            pred_dist = net(img_1, img_2)

            if args.type == 'weighted_l1':
                loss = F.binary_cross_entropy(pred_dist, dist)
            else:
                loss = lowshot.siamese.contrastive_loss(pred_dist, dist, margin=args.margin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        print(f'Loss: {epoch_loss:.2f}', end=' ')

        val_acc = val_tester.test(net.eval(), n_trys=300)
        print(f'Val acc: {val_acc:.2f}')
        if val_acc > best_acc:
            print('Saving model!')
            best_acc = val_acc
            torch.save(net, args.save_path)

            no_improv_counter = 0
        else:
            no_improv_counter += 1

        if no_improv_counter >= args.early_stopping:
            print('Early stopping!')
            break

        epoch_loss = 0.

    print('Loading best model...')
    net = torch.load(args.save_path)

    test_acc = test_tester.test(net.eval(), n_trys=300)
    print(f'Test acc: {test_acc:.2f}')


if __name__ == '__main__':
    lowshot.utils.set_random_seed()

    args = parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    args.save_path = os.path.join(args.save_path, f'siamese_{args.type}.pth')

    train(args)
