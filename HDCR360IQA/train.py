import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random
from datetime import datetime

from torch.cuda.amp import autocast, GradScaler

from torchvision import transforms
from torch.utils.data import DataLoader

from swin.swin_cviqd import HDCR360iq

from config import Config
from utils_tools.process_oiqa import ToTensor, RandHorizontalFlip, Normalize
from utils_tools.process_oiqa import split_dataset_cviqd ,split_dataset_oiqa,split_dataset_iqaodi

from torch.utils.tensorboard import SummaryWriter
from load_train_t import train_oiqa, eval_oiqa 


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['ROC_VISIBLE_DEVICES'] = '0'

os.environ['HIP_LOG_LEVEL'] = '3'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_logging(config):
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )

def print_metrics(epoch, phase, loss, srcc, plcc, mse, epoch_time):
    if phase == 'Train':
        color = '\033[92m'
    else:
        color = '\033[94m'
    reset_color = '\033[0m'
    print(
        f"\n{color}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch [{epoch + 1:3d}/{config.n_epoch:3d}] - {phase} Results {reset_color}")
    print("-" * 80)
    print(f"Loss:        {loss:.6f}")
    print(f"SRCC:        {srcc:.4f}")
    print(f"PLCC:        {plcc:.4f}")
    print(f"mse:        {mse:.4f}")
    print(f"Time Cost:   {epoch_time:.2f}s ({epoch_time / 60:.2f}min)")
    print("-" * 80 + "\n")

if __name__ == '__main__':
    cpu_num = 8
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    config = Config({
        "dataset_name": "iqaodi",
        "cviqd_train_dis_path": "/public/home/huu147/mamba/cviqd/cviqd",
        "cviqd_dis_label": "/public/home/huu147/mamba/cviqd/CVIQD_dis_label.txt",
        "cviqd_ref_label": "/public/home/huu147/mamba/cviqd/CVIQD_ref_label.txt",
        
        "oiqa_train_dis_path":"/public/home/huu147/mamba/data/OIQA/OIQA/distorted_images",
        "oiqa_dis_label": "/public/home/huu147/mamba/data/OIQA/oiqa_dis_label.txt",
        "oiqa_ref_label": "/public/home/huu147/mamba/data/OIQA/oiqa_ref_label.txt",

        "iqaodi_train_dis_path": "/public/home/huu147/mamba/data/all_ref_test_img",
        "iqaodi_dis_label": "/public/home/huu147/mamba/data/IQA_ODI/iqaodi_ref_dis_label.txt",
        
        "batch_size": 8,
        "learning_rate": 2e-5,
        "weight_decay": 1e-5,
        "n_epoch": 120,
        "val_freq": 1,
        "num_workers": 8,
        "split_seed": 20,
        "num_layers": 6,
        "viewport_nums": 5,  # N
        "embed_dim": 224,
        "dab_layers": 3,
        "start_points": [[0, 0], [0, 0], [0, 0]],  # M
        "viewport_size": (224, 224),
        "fov": [110, 110],

        "model_weight_path": "/public/home/huu147/mamba/swin/swin_tiny_patch4_window7_224.pth",
        "model_name": "exp1-iqaodi_seed0_swin_dtk",
        "type_name": "iqaodi",
        "ckpt_path": "./outputc/models/",
        "log_path": "./outputc/log/",
        "tensorboard_path": "./outputc/tensorboard/",
        "warmup_epochs":5
    })

    config.log_file = config.model_name + ".log"
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.type_name, config.model_name)
    config.ckpt_path = os.path.join(config.ckpt_path, config.type_name, config.model_name)
    config.log_path = os.path.join(config.log_path, config.type_name)

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    if not os.path.exists(config.tensorboard_path):
        os.makedirs(config.tensorboard_path)

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(config.tensorboard_path)
    if config.dataset_name == 'cviqd':
        from data.cviqd.cviqd_label_7 import CVIQD
        train_name, val_name = split_dataset_cviqd(config.cviqd_ref_label, config.cviqd_dis_label,
                                                   split_seed=config.split_seed)
        dis_train_path = config.cviqd_train_dis_path
        dis_val_path = config.cviqd_train_dis_path
        label_train_path = config.cviqd_dis_label
        label_val_path = config.cviqd_dis_label
        Dataset = CVIQD
        
    elif config.dataset_name == 'oiqa':
        from data.OIQA.oiqa_label_9 import OIQA
        train_name, val_name = split_dataset_oiqa(config.oiqa_ref_label, config.oiqa_dis_label, split_seed=config.split_seed)
        dis_train_path = config.oiqa_train_dis_path
        dis_val_path = config.oiqa_train_dis_path
        label_train_path = config.oiqa_dis_label
        label_val_path = config.oiqa_dis_label
        Dataset = OIQA

    elif config.dataset_name == 'iqaodi':
        from data.IQA_ODI.iqaodi_label_6 import IQAODI
        train_name, val_name = split_dataset_iqaodi(config.iqaodi_dis_label, split_seed=config.split_seed)
        dis_train_path = config.iqaodi_train_dis_path
        dis_val_path = config.iqaodi_train_dis_path
        label_train_path = config.iqaodi_dis_label
        label_val_path = config.iqaodi_dis_label
        Dataset = IQAODI
        
    else:
        raise ValueError("No dataset, you need to add this new dataset.")

    train_dataset = Dataset(
        dis_path=dis_train_path,
        txt_file_name=label_train_path,
        list_name=train_name,
        transform=transforms.Compose([Normalize(0.5, 0.5),RandHorizontalFlip(),ToTensor() ]),
        viewport_size=config.viewport_size,
        viewport_nums=config.viewport_nums,
        fov=config.fov,
        start_points=config.start_points
    )
    val_dataset = Dataset(
        dis_path=dis_val_path,
        txt_file_name=label_val_path,
        list_name=val_name,
        transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
        viewport_size=config.viewport_size,
        viewport_nums=config.viewport_nums,
        fov=config.fov,
        start_points=config.start_points
    )
    
    logging.info('number of train scenes: {}'.format(len(train_dataset)))
    logging.info('number of val scenes: {}'.format(len(val_dataset)))
    logging.info('train : val ratio is: {:.4}'.format(len(train_dataset) / len(val_dataset)))
    print(f"\n[Dataset Info]")
    print(f"Train Scenes: {len(train_dataset)}")
    print(f"Val Scenes:   {len(val_dataset)}")
    print(f"Train:Val:    {len(train_dataset) / len(val_dataset):.4f}")
    print(f"Batch Size:   {config.batch_size}")
    print(f"Total Epochs: {config.n_epoch}\n")
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False
    )


    net = HDCR360iq()
    if torch.cuda.is_available():
        net = net.cuda()

    scaler = GradScaler(enabled=torch.cuda.is_available(),
                        init_scale=2.**16,
                        growth_factor=2.0,
                        backoff_factor=0.5)

    logging.info(f"AMP Enabled: {scaler.is_enabled()}")
    logging.info(f"CUDA(DCU) Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"DCU Device Name: {torch.cuda.get_device_name(0)}")

    total_params = sum(map(lambda x: x.numel(), net.parameters()))
    trainable_params = sum(map(lambda x: x.numel() if x.requires_grad else 0, net.parameters()))
    print(f"[Model Info]")
    print(f"Total Params: {total_params / 1e6:.2f}M")
    print(f"Trainable Params: {trainable_params / 1e6:.2f}M")
    print(f"Non-trainable Params: {(total_params - trainable_params) / 1e6:.2f}M\n")

    criterion = nn.L1Loss()

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=config.warmup_epochs
    )
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.n_epoch - config.warmup_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[config.warmup_epochs]
    )

    losses, scores = [], []
    best_srocc = 0.0
    best_plcc = 0.0
    main_score = 0.0
    total_train_time = 0.0

    print("=" * 80)
    print(f"Start Training - Model: {config.model_name} | Dataset: {config.dataset_name}")
    print(f"AMP Mode: FP16 (DCU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    for epoch in range(0, config.n_epoch):
        start_time = time.time()

        logging.info('Running training epoch {}'.format(epoch + 1))
        print(
            f"\n[Training] Epoch [{epoch + 1:3d}/{config.n_epoch:3d}] - Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        loss_val, rho_s, rho_p , mse = train_oiqa(epoch, net, criterion, optimizer, train_loader)
        epoch_time = time.time() - start_time
        total_train_time += epoch_time

        writer.add_scalar("Train/Loss", loss_val, epoch)
        writer.add_scalar("Train/SRCC", rho_s, epoch)
        writer.add_scalar("Train/PLCC", rho_p, epoch)
        writer.add_scalar("Train/Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        print_metrics(epoch, 'Train', loss_val, rho_s, rho_p, mse ,epoch_time)

        if (epoch + 1) % config.val_freq == 0:
            logging.info('Starting eval...')
            logging.info('Running val {} in epoch {}'.format(config.dataset_name, epoch + 1))

            val_start_time = time.time()
            val_loss, val_srcc, val_plcc , emse = eval_oiqa(config, epoch, net, criterion, val_loader)
            val_time = time.time() - val_start_time

            logging.info('Eval done...')
            writer.add_scalar("Val/Loss", val_loss, epoch)
            writer.add_scalar("Val/SRCC", val_srcc, epoch)
            writer.add_scalar("Val/PLCC", val_plcc, epoch)
            print_metrics(epoch, 'Val', val_loss, val_srcc, val_plcc, emse, val_time  )


            if val_srcc + val_plcc > main_score:
                main_score = val_srcc + val_plcc
                best_srocc = val_srcc
                best_plcc = val_plcc


                ckpt_name = "best_ckpt.pt"
                model_save_path = os.path.join(config.ckpt_path, ckpt_name)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_srcc': best_srocc,
                    'best_plcc': best_plcc,
                    'main_score': main_score,
                }, model_save_path)

                print(
                    f"\033[91m[Best Model Updated] Epoch {epoch + 1} - Best SRCC: {best_srocc:.4f}, Best PLCC: {best_plcc:.4f}\033[0m")
                logging.info('======================================================================================')
                logging.info(
                    '============================== best main score is {} ================================='.format(
                        main_score))
                logging.info('======================================================================================')
                logging.info(
                    'Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch + 1, best_srocc, best_plcc))

        scheduler.step()
        logging.info('Epoch {} done. Time: {:.2}min | Total Time: {:.2f}min'.format(
            epoch + 1, epoch_time / 60, total_train_time / 60))

    # 训练总结
    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)
    print(f"Total Epochs:    {config.n_epoch}")
    print(f"Total Train Time: {total_train_time:.2f}s ({total_train_time / 3600:.2f}h)")
    print(f"Best Val SRCC:   {best_srocc:.4f}")
    print(f"Best Val PLCC:   {best_plcc:.4f}")
    print(f"Best Main Score: {main_score:.4f}")
    print(f"Best Model Path: {os.path.join(config.ckpt_path, 'best_ckpt.pt')}")
    print("=" * 80)

    logging.info('Training Completed!')
    logging.info(f'Total Train Time: {total_train_time:.2f}s ({total_train_time / 3600:.2f}h)')
    logging.info(f'Best Val SRCC: {best_srocc:.4f}, Best Val PLCC: {best_plcc:.4f}')

    writer.close()