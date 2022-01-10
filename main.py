import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from common.text2id import X_data2id, get_answer_id
import os
import torch
from config.cfg import cfg, path, hyper_roberta
from common.load_data import load_data, tokenizer, data_split_all, generate_template
from model.PromptMask import PromptMask
import torch.optim as optim
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from torch.autograd import Variable
from common.metric import ScorePRF
from common.set_random_seed import setup_seed
import time

if cfg['device'] != 'TPU':
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['gpu_id'])
    device = torch.device(cfg['device'])
else:
    # for TPU
    print('TPU using ....')
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()

acc_array_1, f1_array, acc_array_2 = [], [], []
seeds = [10, 100, 1000, 2000, 4000]
average_acc_1, average_f1, average_acc_2 = 0, 0, 0
for test_id in range(len(seeds)):
    print('~~~~~~~~~~~~~ 第', test_id+1, '次测试 ~~~~~~~~~~~~~~~~~~~')
    setup_seed(seeds[test_id])

    train_X, train_y = load_data(path['train_path'])
    train_X0, train_y0, _, _ = data_split_all(train_X, train_y, 5, cfg['K'])
    test_X, test_y = load_data(path['test_path'])
    test_X, test_y = np.array(test_X), np.array(test_y)

    train_X, train_y = generate_template(train_X0, train_X0, train_y0, train_y0, True)
    test_X, test_y = generate_template(test_X, train_X0, test_y, train_y0)

    train_X, test_X = X_data2id(train_X, tokenizer), X_data2id(test_X, tokenizer)
    train_y, answer_map = get_answer_id(train_y, tokenizer)
    test_y, _ = get_answer_id(test_y, tokenizer)

    train_X, train_y = torch.tensor(train_X), torch.tensor(train_y)
    test_X, test_y = torch.tensor(test_X), torch.tensor(test_y)

    train_data = TensorDataset(train_X, train_y)
    test_data = TensorDataset(test_X, test_y)

    loader_train = DataLoader(
        dataset=train_data,
        batch_size=cfg['train_batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    loader_test = DataLoader(
        dataset=test_data,
        batch_size=cfg['test_batch_size'],
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    net = PromptMask()
    net = net.to(device)

    def change_lr(optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    current_lr = cfg['learning_rate']
    if cfg['optimizer'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=cfg['learning_rate'])
    elif cfg['optimizer'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=cfg['learning_rate'], weight_decay=1e-3)
    elif cfg['optimizer'] == 'AdamW':
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg['learning_rate'], eps=1e-8)
        num_warmup_steps = 0
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=len(loader_train) // 1 * cfg['epoch'])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True,
    #                                                        threshold=0.0001,
    #                                                        threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    epoch = cfg['epoch']
    print(cfg)
    print(hyper_roberta)

    for i in range(epoch):
        # if i > 5:
        #     current_lr *= 0.95
        #     change_lr(optimizer, current_lr)

        print('-------------------------   training   ------------------------------')
        time0 = time.time()
        batch = 0
        ave_loss, sum_acc = 0, 0
        for batch_x, batch_y in loader_train:
            net.train()
            batch_x, batch_y = Variable(batch_x).long(), Variable(batch_y).long()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)


            output = net(batch_x)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            if cfg['device'] == 'TPU':
                xm.optimizer_step(optimizer)
                xm.mark_step()
            else:
                optimizer.step()  # 更新权重
            if cfg['optimizer'] == 'AdamW':
                scheduler.step()
            optimizer.zero_grad()  # 清空梯度缓存
            ave_loss += loss
            batch += 1

            if batch % 25 == 0:
                print('epoch:{}/{},batch:{}/{},time:{}, loss:{},learning_rate:{}'.format(i + 1, epoch, batch,
                                                                                         len(loader_train),
                                                                                         round(time.time() - time0, 4),
                                                                                         loss,
                                                                                         optimizer.param_groups[
                                                                                             0]['lr']))
        # scheduler.step(ave_loss)
        print('------------------ epoch:{} ----------------'.format(i + 1))
        print('train_average_loss{}'.format(ave_loss / len(loader_train)))
        print('============================================'.format(i + 1))

        time0 = time.time()
        if (i + 1) % epoch == 0:
            score = ScorePRF()
            label_out, label_y = [], []
            label_out_2, label_y_2 = [], []
            print('-------------------------   test   ------------------------------')
            sum_acc, num = 0, 0
            # torch.save(net.state_dict(), 'save_model/params' + str(i + 1) + '.pkl')
            for batch_x, batch_y in loader_test:
                net.eval()
                batch_x, batch_y = Variable(batch_x).long(), Variable(batch_y).long()
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                with torch.no_grad():
                     output = net(batch_x)

                _, pred = torch.max(output, dim=1)

                pred = pred.cpu().detach().numpy()
                batch_y = batch_y.cpu().detach().numpy()

                for j in range(pred.shape[0]):
                    label_out.append(pred[j])
                    label_y.append(batch_y[j])

                votes = np.zeros(5)
                for type_id in range(5):
                    for j in range(type_id * cfg['K'], (type_id + 1) * cfg['K']):
                        if pred[j] == answer_map[1]:
                            votes[type_id] += 1

                    if batch_y[type_id * cfg['K'] + 1] == answer_map[1]:
                        label_y_2.append(type_id)

                label_out_2.append(np.argmax(votes))

            label_out = np.array(label_out)
            label_y = np.array(label_y)
            score.cal_tp_fp_fn(label_y, label_out, 1)

            p1, r1, f1 = score.cal_label_f1()
            acc_1 = (np.sum(label_y == label_out)) / len(label_y)
            acc_2 = (np.sum(label_y_2 == label_out_2)) / len(label_y_2)
            print('------------------ epoch:{} ----------------'.format(i + 1))
            print('test_acc1:{}, time:{}'.format( round(acc_1, 4), time.time()-time0))
            print('============================================'.format(i + 1))
            average_acc_1 += acc_1 * 100
            average_f1 += f1 * 100
            average_acc_2 += acc_2 * 100
            acc_array_1.append(acc_1 * 100)
            f1_array.append(f1 * 100)
            acc_array_2.append(acc_2 * 100)


average_acc_1 /= 5
average_acc_2 /= 5
average_f1 /= 5

acc_array_1 = np.array(acc_array_1)
acc_array_2 = np.array(acc_array_2)
f1_array = np.array(f1_array)
print('average_acc_1:{}, std:{}'.format(round(average_acc_1, 4), round(np.std(acc_array_1, ddof=1), 4)))
print('average_f1:{}, std:{}'.format(round(average_f1, 4), round(np.std(f1_array, ddof=1), 4)))
print('average_acc_2:{}, std:{}'.format(round(average_acc_2, 4), round(np.std(acc_array_2, ddof=1), 4)))
