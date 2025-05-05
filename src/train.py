import torch
import os
import numpy as np

from models.model import create_model, load_model, save_model
from datasets.dataset_factory import get_dataset
from opts import opts
from trains.train_factory import train_factory
from utils.logger import Logger

from pathlib import Path

def main(opt):
    torch.manual_seed(opt.seed)
    opt.device = torch.device('cuda')

    # Create dataset
    Dataset = get_dataset(opt.dataset, opt.task)

    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    # Dataset.__getitem__(Dataset(opt, 'train'), index=0)

    # Create model
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt.wavelet_setting).to(opt.device)

    if opt.task == 'eiss':
        opt.eiss_alpha = torch.nn.Parameter(torch.tensor(opt.eiss_alpha))
        optimizer = torch.optim.Adam(list(model.parameters()) + [opt.eiss_alpha], opt.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step, opt.load_dino)

    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.device)

    def custom_collate_fn(batch):
        ret_batch = {key: torch.stack([torch.tensor(d[0][key]) for d in batch]) for key in batch[0][0]}
        target_batch = [d[1] for d in batch]
        return ret_batch, target_batch

    dataset_train = Dataset(opt, 'train')
    dataset_val = Dataset(opt, 'val')
    if opt.task == 'selfdet':
        cache_dir = os.path.join(opt.save_dir, 'cache')
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        dataset_train.init_selfdet(cache_dir=cache_dir, max_prop=30, strategy=opt.selfdet_strategy)
        dataset_val.init_selfdet(cache_dir=cache_dir, max_prop=30, strategy=opt.selfdet_strategy)
    elif opt.task == 'eiss':
        dataset_train.init_eiss(opt.eiss_strategy)
        dataset_val.init_eiss(opt.eiss_strategy)
    elif opt.task == 'TC':
        dataset_train.init_TC()
        dataset_val.init_TC()
    elif opt.task == 'simclr':
        dataset_train.init_simclr()
        dataset_val.init_simclr()
        opt.n_views = 2

    # Create dataloader
    print('Setting up data...')
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        #num_workers=1,
        pin_memory=True,
        collate_fn = custom_collate_fn
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=opt.batch_size,
        shuffle=True,
        # TODO: Enable data loader multithread
        #num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    logger = Logger(opt)

    print('Starting training...')
    best = 1e10

    if opt.resume and os.path.exists(os.path.join(opt.save_dir, 'best.npy')):
        best = np.load(os.path.join(opt.save_dir, 'best.npy'))

    if opt.resume:
        with torch.no_grad():
            log_dict_val, preds, stats = trainer.val(start_epoch, train_loader)
        for k, v in log_dict_val.items():
            logger.scalar_summary('val_{}'.format(k), v, start_epoch)
            logger.write('{} {:8f} | '.format(k, v))
        logger.write(str(stats))

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ , _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds, stats = trainer.val(epoch, val_loader)

            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            logger.write(str(stats))

            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]

                np.save(os.path.join(opt.save_dir, 'best.npy'), best)

                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
    logger.close()


if __name__ == '__main__':
    opt = opts()
    opt= opt.parse()
    main(opt)