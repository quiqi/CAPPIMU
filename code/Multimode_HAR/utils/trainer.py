import sklearn
import numpy as np
import os
from copy import deepcopy
from models.backbones import *
from utils.util import ConfusionMatrix, save_record
from data_preprocess import data_preprocess_ucihar, data_preprocess_insole, data_preprocess_pamap2
from tqdm import tqdm
import fitlog

# create directory for saving models and plots
global save_dir
save_dir = 'results/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
global plot_dir_name
plot_dir_name = 'plot/'
if not os.path.exists(plot_dir_name):
    os.makedirs(plot_dir_name)


def train(train_loaders, test_loader, model, DEVICE, optimizer, scheduler, criterion, args):
    best_model = None
    best_metric = 0
    train_loss = []
    train_metrics_batch = []
    epoch_train_losses = []
    epoch_train_metric = []
    epoch_test_losses = []
    epoch_test_metric = []
    params = {
        'best_model': best_model,
        'epochs': [],
        'train_loss': [],
        'test_loss': [],
        'lr': [],
        'train_metric': [],
        'test_metric': []
    }
    for epoch in range(args.n_epoch):
        model.train()
        for i, train_loader in enumerate(train_loaders):
            for idx, (sample, target, domain) in enumerate(tqdm(train_loader)):
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                optimizer.zero_grad()
                out = model(sample)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())
                train_metrics_batch.append(
                    sklearn.metrics.f1_score(target.cpu(), torch.argmax(out, dim=1).cpu(), average='micro'))

        p = optimizer.state_dict()['param_groups'][0]['lr']  # 学习率
        params['lr'].append(p)
        params['epochs'].append(epoch)
        # scheduler.step()

        train_loss_avg = np.mean(train_loss)
        train_metric_avg = np.mean(train_metrics_batch)
        epoch_train_losses.append(train_loss_avg)
        epoch_train_metric.append(train_metric_avg)

        test_loss_avg, test_metric_avg, best_metric, best_model = \
            test(model, test_loader, criterion, best_metric, best_model, DEVICE)

        epoch_test_losses.append(test_loss_avg)
        epoch_test_metric.append(test_metric_avg)

        train_f1_score = float(train_metric_avg) * 100.0
        test_f1_score = float(test_metric_avg) * 100.0

        print(f'\nFold: {args.fold + 1}/{args.k_fold} | Epoch : {epoch + 1}\n')
        print(f'Train Loss : {train_loss_avg:.4f}\t | \tTrain F1_Score : {train_f1_score:2.4f}\n')
        print(f'test Loss : {test_loss_avg:.4f}\t | \ttest F1_Score : {test_f1_score:2.4f}\n')

        if test_metric_avg == best_metric:
            print('update')
            model_dir = save_dir + args.model_name + '.pt'
            print('Saving models at {} epoch to {}'.format(epoch + 1, model_dir))
            # torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
            #            model_dir)


    params['train_loss'] = epoch_train_losses
    params['train_metric'] = epoch_train_metric
    params['test_loss'] = epoch_test_losses
    params['test_metric'] = epoch_test_metric
    params['best_model'] = best_model

    return params


def test(model, test_loader, criterion, best_metric, best_model,  DEVICE):
    model.eval()
    test_loss = []
    test_metrics_batch = []

    with torch.no_grad():
        for idx, (sample, target, domain) in enumerate(tqdm(test_loader)):
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            out = model(sample)
            loss = criterion(out, target)
            test_loss.append(loss.item())
            test_metrics_batch.append(
                sklearn.metrics.f1_score(target.cpu(), torch.argmax(out, dim=1).cpu(), average='micro'))

        test_loss_avg = np.mean(test_loss)
        test_metrics_avg = np.mean(test_metrics_batch)
        if test_metrics_avg > best_metric:
            best_metric = test_metrics_avg
            best_model = deepcopy(model.state_dict())  # 保存训练epoch中指标最好的参数模型

    model.train()

    return test_loss_avg, test_metrics_avg, best_metric, best_model


def evaluate(test_loader, best_model, DEVICE, criterion, args):
    model, _ = setup_model_optm(args, DEVICE)
    model.load_state_dict(best_model)
    confusion = ConfusionMatrix(num_classes=args.n_class, labels=args.labels, device=args.device)
    with torch.no_grad():
        model.eval()
        test_loss = []
        test_metrics_batch = []
        for idx, (sample, target, domain) in enumerate(tqdm(test_loader)):
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            out = model(sample)
            loss = criterion(out, target)
            test_loss.append(loss.item())
            test_metrics_batch.append(sklearn.metrics.f1_score(target.cpu(), torch.argmax(out, dim=1).cpu(),
                                                               average='micro'))
            outputs = torch.argmax(out, dim=1)
            confusion.update(outputs.to("cpu").numpy(), target.to("cpu").numpy())

        test_loss_avg = np.mean(test_loss)
        test_metrics_avg = np.mean(test_metrics_batch)
        test_f1_score = float(test_metrics_avg) * 100.0

    print(f'Fold: {args.fold + 1}/{args.k_fold} | Evaluate:\n')
    print(f'Final Test Loss : {test_loss_avg:.4f}\t | \tFinal Test F1_Score : {test_f1_score:2.4f}\n')
    save_record(f"device:{args.device} | train_device:{args.train_device} | train_device:{args.test_device}",
                args.logdir + args.model_name)
    save_record(f'Fold: {args.fold + 1}/{args.k_fold} | Evaluate:', args.logdir + args.model_name)
    save_record(f"Final Test Loss : {test_loss_avg:.4f}\t | \tFinal Test F1_Score : {test_f1_score:2.4f}",
                args.logdir + args.model_name)
    acc = confusion.summary()
    save_record(f"Final Test accuracy is {acc}\n", args.logdir + args.model_name)
    # confusion.plot()


def setup_dataloaders(args):
    if args.dataset == 'ucihar':
        args.n_feature = 9
        args.len_sw = 128
        args.n_class = 6
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain == '0'
        train_loaders, test_loader = \
            data_preprocess_ucihar.prep_ucihar(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int(args.len_sw * 0.5))

    if args.dataset == 'pamap2':
        args.n_feature = 52
        args.len_sw = 170
        args.n_class = 12
        args.shape = (1, 170, 52)
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain = '101'
        train_loaders, test_loader = \
            data_preprocess_pamap2.prep_pamap2(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int(args.len_sw * 0.5))

    if args.dataset == 'insole':
        args.n_class = 21
        if args.device == 'insole':
            args.n_feature = 16
        elif len(args.device) >= 8:
            args.n_feature = 25
        else:
            args.n_feature = 9

        if args.cases == 'random_device':
            args.n_feature = 9
            args.train_device, args.test_device = get_device_data(args.fold)

        source_domain_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10','11', '12', '13', '14', '15', '16',
                              '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
        # train_domain = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
        #                 '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']
        # test_domain = ['30']
        train_domain, test_domain = get_k_fold_data(args.k_fold, args.fold, source_domain_list)
        train_loaders, test_loader = \
            data_preprocess_insole.prep_insole(args, SLIDING_WINDOW_LEN=args.len_sw,
                                               SLIDING_WINDOW_STEP=int(args.len_sw * 0.5), train_user=train_domain,
                                               test_user=test_domain)

    return train_loaders, test_loader

  
def setup_model_optm(args, DEVICE):
    # set up backbone network
    if args.backbone == 'FCN':
        model = FCN(n_channels=args.n_feature, n_timesteps=args.len_sw, n_classes=args.n_class)
    elif args.backbone == 'DCL':
        model = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=128)
    elif args.backbone == 'TPN':
        model = TPN(in_channels=args.n_feature, n_classes=args.n_class)
    elif args.backbone == 'Transformer':
        model = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4,
                            heads=4, mlp_dim=64, dropout=0.1)
    elif args.backbone == 'resnet':
        model = resnet()
    elif args.backbone == 'DCLS':
        model = DC_shallow_LSTM()
    elif args.backbone == 'TA':
        model = CNN_TA()
    else:
        NotImplementedError

    model = model.to(DEVICE)

    if args.optimizer == 'sgd':
        optimizers = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizers = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

    return model, optimizers


def setup(args, DEVICE):
    model, optimizers = setup_model_optm(args, DEVICE)
    criterion = nn.CrossEntropyLoss()

    args.model_name = args.backbone + '_cases-' + args.cases + '_lr' + str(args.lr) + '_bs' + str(
        args.batch_size) + '_sw' + str(args.len_sw)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers, T_max=args.n_epoch, eta_min=0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers, gamma=0.85)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizers, step_size=100, gamma=0.85)

    return model, optimizers, scheduler, criterion


def get_k_fold_data(k, i, X):
    """
    返回第i折训练的训练集的志愿者编号和测试集志愿者编号,X_train_user为训练数据,X_test_user为验证数据
    k: k折交叉验证-------> 30
    i: 第i折作测试集
    X: source_domain_list
    """
    assert k > 1 and i < k
    fold_size = len(X) // k  # 6 每份的个数:数据总条数/折数（组数）
    X_train = []
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        # idx 为每组 valid
        X_part = X[idx]
        if j == i:  # 第i折作valid
            X_test = X_part
        elif X_train is None:
            X_train = X_part
        else:
            X_train = np.concatenate([X_train, X_part])

    return X_train, X_test


def get_device_data(fold_iter):
    assert fold_iter < 81
    # device = ['insole_head', 'insole_arm_l', 'insole_arm_r', 'insole_wrist_l', 'insole_wrist_r',
    #           'insole_chest', 'insole_knee_l', 'insole_knee_r', 'insole_pocket']
    device = ['head', 'arm_l', 'arm_r', 'wrist_l', 'wrist_r',
              'chest', 'knee_l', 'knee_r', 'pocket']
    train_index = int(fold_iter / 9)
    test_index = fold_iter % 9

    return device[train_index], device[test_index]
