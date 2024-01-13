from utils.trainer import *
from utils.args_parser import *
from utils.util import *
import torch
from datetime import datetime


if __name__ == '__main__':
    args = init_parameters('--dataset insole --device insole_wrist_r --backbone TA --n_epoch 100 ' 
                           '--batch_size 128 --split_ratio 0.3 --lr 0.01 --len_sw 300'.split())

    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('device:', DEVICE, 'dataset:', args.dataset, 'cases:', args.cases)
    print('Architecture is {}'.format(args.backbone))
    training_start = datetime.now()

    if args.cases == 'subject' or args.cases == 'random_device':
        fold_iter = args.k_fold
        # fold_iter = 1  // 单步执行时使用
    else:
        fold_iter = 1

    for i in range(fold_iter):
        args.fold = i

        seed_torch(5)
        train_loaders, test_loader = setup_dataloaders(args)
        model, optimizers, scheduler, criterion = setup(args, DEVICE)
        params = train(train_loaders, test_loader, model, DEVICE, optimizers, scheduler, criterion, args)
        evaluate(test_loader, params['best_model'], DEVICE, criterion, args)

    training_end = datetime.now()
    training_time = training_end - training_start
    print(f"Training time is : {training_time}")

    # pic(params['train_loss'], params['test_loss'], params['train_metric'],
    #     params['test_metric'], pic_name=args.dataset)
