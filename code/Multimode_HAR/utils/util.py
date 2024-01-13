import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import torch
import random
import os


def save_record(content, path, black_line=True):
    # print(type(content[0]))
    if not isinstance(content, list):
        if not isinstance(content, str):
            content = str(content)
        content = [content]

    txt_object = open(path, "a+")
    if black_line:
        txt_object.write("\n")
    txt_object.write('\n'.join(content) + '\n')
    txt_object.close()


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def args_contains(args, name, default):
    if hasattr(args, name):
        return getattr(args, name)
    else:
        return default


def pic(train_loss, test_loss, train_metrics, test_metrics, pic_name):
    """画图"""
    plt.figure(figsize=(8, 8))
    plt.title('result')  # 折线图标题
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
    plt.xlabel('epoch')  # x轴标题
    plt.ylabel('y')  # y轴标题
    plt.plot(train_loss, linewidth=2.0, linestyle='-', label='train_loss')  # 绘制折线图，添加数据点，设置点的大小
    plt.plot(test_loss, linewidth=2.0, linestyle='-', label='test_loss')
    plt.plot(train_metrics, linewidth=2.0, linestyle='-', label='train_metrics')
    plt.plot(test_metrics, linewidth=2.0, linestyle='-', label='test_metrics')
    plt.grid(which='major', axis='both')  # 显示网格
    plt.legend()  # 设置折线名称
    plt.savefig('plot/{0}.png'.format(pic_name))
    plt.show()


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list, device: str, normalize=True):
        self.matrix = np.zeros((num_classes, num_classes), dtype="float32")
        self.num_classes = num_classes
        self.labels = labels
        self.device = device
        self.normalize = normalize

    def update(self, preds, labels):
        for t, p in zip(labels, preds):
            self.matrix[t, p] += 1

    def getMatrix(self, normalize=True):
        """
        if normalize=True, matrix is in percentage form
        if normalize=False, matrix is in numerical  form
        """
        if normalize:
            per_sum = self.matrix.sum(axis=1)
            for i in range(self.num_classes):
                self.matrix[i] = (self.matrix[i] / per_sum[i])
                self.matrix = np.around(self.matrix, 2)
                self.matrix[np.isnan(self.matrix)] = 0
        return self.matrix

    def plot(self):
        matrix = self.getMatrix(self.normalize)
        # print("---------Confusion Matrix--------")
        # print(matrix)
        plt.figure(figsize=(15, 15))
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=90, fontsize=12)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels, fontsize=12)
        # 显示colorbar
        plt.colorbar(fraction=0.046, extend='both')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = float(format('%.2f' % matrix[y, x]))
                plt.text(x, y, info,
                         fontsize=10,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()  # 保证图不重叠
        plt.savefig('plot/' + self.device + '_confusion_matrix.svg', format='svg')
        plt.show()

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is {}\n".format(acc))

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "F1_score"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FN = np.sum(self.matrix[i, :]) - TP
            FP = np.sum(self.matrix[:, i]) - TP
            # TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            F1_score = round(2 * Precision * Recall / (Precision + Recall), 3) if Precision + Recall != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, F1_score])
        print(table)

        return acc
