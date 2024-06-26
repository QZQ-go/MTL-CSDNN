import optuna
import pandas
import os
import matplotlib.pyplot as plt
import torch
from typing import Union
from data_loader import DiseaseDataSet
from net import MultiTaskDnn, disease_name
from torch.utils.data import random_split, DataLoader
from utils import init_console_and_file_log
from tqdm import tqdm
from utils.pytorch_model_kit import TestMetricsRecoder, MetricsRecoder
from config import device
from itertools import product

torch.multiprocessing.set_sharing_strategy('file_system')


def my_task(trial: Union[optuna.trial.Trial, None],
            use_k_fold=False,
            k=10,
            fold_num=1,
            prefix=None,
            random_seed=None):
    if trial:
        BATCH_SIZE = trial.suggest_categorical('BATCH_SIZE', [256, 512, 1024])
        POS_WEIGHT = trial.suggest_int('POS_WEIGHT', 10, 30)
        NEG_WEIGHT = trial.suggest_int('NEG_WEIGHT', 0, 10)
        LR = trial.suggest_float("LR", 1e-5, 1e-1, log=True)
        momentum = trial.suggest_float('momentum', 0.05, 0.2, step=0.01)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        step_size = trial.suggest_int('step_size', 5, 20)
        gamma = trial.suggest_float('gamma', 0.1, 0.8, step=0.1)
    else:
        # BATCH_SIZE = 256
        # POS_WEIGHT = 30
        # NEG_WEIGHT = 10
        # LR = 1e-2
        # momentum = 0.08
        # weight_decay = 1e-4
        # step_size = 4
        # gamma = 0.7

        BATCH_SIZE = 1024
        POS_WEIGHT = 30
        NEG_WEIGHT = 8
        LR = 0.06
        momentum = 0.15
        weight_decay = 1.1e-4
        step_size = 8
        gamma = 0.7
    if not random_seed:
        random_seed = 5

    # 全局参数
    # EPOCHS = 52
    # AIMED_ACC = 0.7
    # AIMED_REC = 0.7
    # NUM_WORK = 6
    # PIN_MEMORY = True

    EPOCHS = 52
    AIMED_ACC = 0.7
    AIMED_REC = 0.7
    NUM_WORK = 0
    PIN_MEMORY = False

    if trial:
        PREFIX = f'{trial.number}_mt_dnn'
    elif use_k_fold:
        PREFIX = f'k{k}_{fold_num}_mt_dnn'
    elif prefix:
        PREFIX = prefix
    else:
        PREFIX = 'mt_dnn'

    if use_k_fold:
        if not os.path.exists("k_fold"):
            os.mkdir('k_fold')
        target_dir = f'k_fold/k{k}_{fold_num}'
        os.mkdir(target_dir)
    elif trial:
        if not os.path.exists("trial"):
            os.mkdir('trial')
        target_dir = f'trial/trial_{trial.number}'
        os.mkdir(target_dir)
    else:
        t = 'files/'
        if not os.path.exists(t):
            os.mkdir(t)
        target_dir = os.path.join(t, f'{PREFIX}/')
        os.mkdir(target_dir)

    LOGGER_FILE_NAME = os.path.join(target_dir, f'{PREFIX}_train.log')
    MODEL_FILE_NAME = os.path.join(target_dir, f'{PREFIX}_model_para.pt')
    IMAGE_NAME = os.path.join(target_dir, f'{PREFIX}_task.png')
    CSV_NAME = os.path.join(target_dir, f'{PREFIX}_task.csv')
    SINGLE_DISEASE_CSV_NAME = os.path.join(target_dir, f'{PREFIX}_task_single_disease.csv')
    PR_AUC_FILE_NAME = os.path.join(target_dir, f'{PREFIX}_pr_auc.csv')
    METRICS_PATH = os.path.join(target_dir, f'{PREFIX}_metrics.csv')
    SINGLE_DISEASE_METRICS_PATH = os.path.join(target_dir, f'{PREFIX}_single_disease_metrics.csv')

    logger = init_console_and_file_log("Trainer", LOGGER_FILE_NAME)
    logger.info(f'use device {device.type}')

    recorder = {f'{tp}_{mtc}': [] for tp, mtc in product(['train', 'val'], ['loss', 'rec', 'acc'])}
    des_recorder = {f'{des}_{tp}_{mtc}': [] for des, tp, mtc in
                    product(disease_name, ['train', 'val'], ['loss', 'rec', 'acc'])}

    best_acc = 0
    best_rec = 0
    best_value = 0

    # 定义数据集
    data_set = DiseaseDataSet()

    if use_k_fold:
        k_ratio = round(1 / k, 1)
        before = round(k_ratio * (fold_num - 1), 1)
        k_fold = k_ratio
        after = round(1 - before - k_fold, 1)

        bf_dataset, test_dataset, af_dataset = random_split(
            dataset=data_set,
            lengths=[before, k_fold, after],
            generator=torch.Generator().manual_seed(random_seed)
        )

        rest_dataset = bf_dataset + af_dataset
        train_dataset, val_dataset = random_split(
            dataset=rest_dataset,
            lengths=[0.9, 0.1],
            generator=torch.Generator().manual_seed(random_seed)
        )
    else:
        train_len = int(round(len(data_set) * 0.8, 0))
        test_len = int(round(len(data_set) * 0.1, 0))
        train_len += len(data_set) - train_len - test_len * 2

        train_dataset, val_dataset, test_dataset = random_split(
            dataset=data_set,
            lengths=[train_len, test_len, test_len],
            generator=torch.Generator().manual_seed(random_seed)
        )

    data_set_dict = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    data_loader_dict = {k: DataLoader(data_set_dict[k],
                                      shuffle=True,
                                      batch_size=BATCH_SIZE,
                                      num_workers=NUM_WORK,
                                      pin_memory=PIN_MEMORY) for k in data_set_dict.keys()}

    # 定义模型等
    model = MultiTaskDnn()
    model.to(device)

    # 定义损失函数等
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=LR,
                                momentum=momentum,
                                weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=step_size,
                                                gamma=gamma)

    for epoch in range(EPOCHS):
        logger.info(f"epoch {epoch}")
        torch.cuda.empty_cache()  # 释放显存

        for mode in ["train", "val"]:
            model.train() if mode == "train" else model.eval()

            t_metrics = MetricsRecoder()
            dse_t_metrics = {dse: MetricsRecoder() for dse in disease_name}

            target_data_loader = data_loader_dict[mode]
            for fold_num, data in enumerate(tqdm(target_data_loader)):
                if mode == "train":
                    optimizer.zero_grad()

                label: dict = data[1]
                inputs: dict = data[0]
                for k in inputs.keys():
                    inputs[k] = inputs[k].to(device)
                    label[k] = label[k].float().to(device)

                outputs: dict = model(inputs)
                for k in outputs.keys():
                    outputs[k] = outputs[k].view(outputs[k].shape[0])

                loss_list = []
                for k in label.keys():
                    # 计算loss
                    # 第二种权重赋值方式，当是正例并且是判断错误的时候才增加权重
                    weight_tensor = torch.where(
                        (label[k] == 1) & (outputs[k].round() == 0), POS_WEIGHT, 1
                    )
                    weight_tensor = torch.where(
                        (label[k] == 1) & (outputs[k].round() == 1), NEG_WEIGHT, weight_tensor
                    )

                    loss_func.weight = weight_tensor.to(device)
                    loss = loss_func(outputs[k], label[k])
                    loss_list.append(loss)

                    t_metrics.load(label[k], outputs[k], loss)
                    if not trial:
                        dse_t_metrics[k].load(label[k], outputs[k], loss)

                final_loss: torch.Tensor = sum(loss_list)

                if mode == "train":
                    final_loss.backward()
                    optimizer.step()

            # 记录该轮训练产生的数据
            acc, rec, total_loss = t_metrics.get_metrics()
            if not trial:
                dse_res_dict = {dse: dse_t_metrics[dse].get_metrics() for dse in disease_name}

            recorder[f'{mode}_loss'].append(total_loss)
            recorder[f'{mode}_rec'].append(rec)
            recorder[f'{mode}_acc'].append(acc)

            logger.info(f"{mode} loss {total_loss}")
            logger.info(f"{mode} acc {acc}")
            logger.info(f"{mode} rec {rec}")

            if not trial:
                for des in disease_name:
                    des_recorder[f'{des}_{mode}_acc'].append(dse_res_dict[des][0])
                    des_recorder[f'{des}_{mode}_rec'].append(dse_res_dict[des][1])
                    des_recorder[f'{des}_{mode}_loss'].append(dse_res_dict[des][2])

        scheduler.step()

        # 每个世代对结果进行制图
        fig = plt.figure()
        # 在第1，2，4的位置添加面板
        ax_loss, ax_acc, ax_rec = fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(224)

        ax_loss.set(title='Loss')
        ax_acc.set(title='Acc')
        ax_rec.set(title='Rec')

        for _type in ["train", "val"]:
            ax_loss.plot(recorder[f'{_type}_loss'], label=_type)
            ax_acc.plot(recorder[f'{_type}_acc'], label=_type)
            ax_rec.plot(recorder[f'{_type}_rec'], label=_type)

        plt.legend()
        plt.savefig(IMAGE_NAME)

        # 将训练记录保存为一个csv文件
        df = pandas.DataFrame(recorder)
        df.to_csv(CSV_NAME)
        if not trial:
            df = pandas.DataFrame(des_recorder)
            df.to_csv(SINGLE_DISEASE_CSV_NAME)

        # 如果精确度和召回率达到预期，则开始储存模型
        val_acc = recorder[f'val_acc'][-1]
        val_rec = recorder[f'val_rec'][-1]

        if val_acc > AIMED_ACC or val_rec > AIMED_REC:
            if val_acc > best_acc or val_rec > best_rec:
                best_acc = val_acc
                best_rec = val_rec
                torch.save(model.state_dict(), MODEL_FILE_NAME)

        # 每10次运行一次测试集
        if epoch % 10 == 0 and os.path.exists(MODEL_FILE_NAME):
            logger.info('start testing!')  # 当模型训练结束，加载最优参数进行结果测试
            test_model = MultiTaskDnn()
            test_model.load_state_dict(torch.load(MODEL_FILE_NAME))
            test_model.to(device)
            test_model.eval()

            metrics_dict = {ds: TestMetricsRecoder() for ds in disease_name}
            metrics_dict.update({'all': TestMetricsRecoder()})

            for fold_num, data in enumerate(tqdm(data_loader_dict['test'])):
                label: dict = data[1]
                inputs: dict = data[0]
                for k in inputs.keys():
                    inputs[k] = inputs[k].to(device)
                    label[k] = label[k].float().to(device)

                outputs: dict = test_model(inputs)

                for k in label.keys():
                    metrics_dict['all'].load(label[k], outputs[k], None)
                    if not trial:
                        metrics_dict[k].load(label[k], outputs[k], None)

            all_metrics = metrics_dict['all'].get_metrics(PR_AUC_FILE_NAME)
            if not trial:  # 如果在进行参数搜索，那么没必要进行测试
                pandas.DataFrame([all_metrics]).T.to_csv(METRICS_PATH)
                metrics_dict.pop('all')
                res = {key: metrics_dict[key].get_metrics(os.path.join(target_dir, f'{PREFIX}_{key}_pr_auc.csv')) for key in metrics_dict.keys()}
                pandas.DataFrame(res).to_csv(SINGLE_DISEASE_METRICS_PATH)

            if all_metrics['prc_auc'] > best_value:
                best_value = all_metrics['prc_auc']

            if trial:
                trial.report(all_metrics['prc_auc'], epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned

            logger.info(f'test metrics: {all_metrics}')

    return best_value


def trial_task():
    study = optuna.create_study(
            storage="sqlite:///multi_task.sqlite3",  # 数据文件存放的地址
            study_name="multi_task",  # 需要指定学习任务的名字，该名字就是数据文件的名字
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
            load_if_exists=True)
    study.optimize(my_task, n_trials=40)

    print(f'best params:{study.best_params}, best value:{study.best_value}, best trial:{study.best_trial}')


if __name__ == '__main__':
    # for i in range(6, 10, 1):
    #     my_task(None, use_k_fold=True, fold_num=i+1)

    my_task(None, use_k_fold=True, fold_num=10)

    # my_task(None, use_k_fold=False)
    # for i in range(10):
    #     pre = f'平行实验随机数{i+1}'
    #     my_task(None, if_trail=False, use_k_fold=False, prefix=pre, random_seed=i+1)
    # trial_task()
