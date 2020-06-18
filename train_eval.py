# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils_fastTextTest import get_time_dif
from tensorboardX import SummaryWriter


# 权重初始化，默认xavier（如果不初始化，则默认的随机权重会特别大，对模型训练造成影响）
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters(): # 迭代网络中所有可训练的参数
        if exclude not in name:  # 排除名字中包含指定关键词的参数（默认为'embedding'）
            if 'weight' in name: # 对权重进行初始化
                if method == 'xavier':
                    nn.init.xavier_normal_(w) # 调用不同的初始化方法
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name: # 对偏置进行初始化
                nn.init.constant_(w, 0)
            else: # 跳过除权重和偏置外的其他参数
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train() # model.train()将启用BatchNormalization和Dropout，相应的，model.eval()则不启用BatchNormalization和Dropout
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) # 指定优化方法

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu() # 从cpu tensor中取出标签数据
                predic = torch.max(outputs.data, 1)[1].cpu() # 返回每一行中最大值的列索引
                train_acc = metrics.accuracy_score(true, predic) # 计算这个batch的分类准确率
                dev_acc, dev_loss = evaluate(config, model, dev_iter) # 计算开发集上的准确率和训练误差
                if dev_loss < dev_best_loss:  # 使用开发集判断模型性能是否提升
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                # 开发集loss超过一定数量的batch没下降，则结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval() # 不启用BatchNormalization和Dropout
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():  # 不追踪梯度
        for texts, labels in data_iter:  # 对数据集中的每一组数据
            outputs = model(texts)  # 使用模型进行预测
            loss = F.cross_entropy(outputs, labels) # 计算模型损失
            loss_total += loss # 累加模型损失
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels) # 记录标签
            predict_all = np.append(predict_all, predic) # 记录预测结果

    acc = metrics.accuracy_score(labels_all, predict_all) # 计算分类误差
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter) # 返回分类误差和平均模型损失
