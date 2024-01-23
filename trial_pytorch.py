import traceback
from pytorch_widedeep.models import TabTransformer, SAINT
from config import *
from data_loader import *
from net import *
from pytorch_task import get_path_params_dict, save_params, get_data_loader
from utils.pack_model_kit import CustomDataset
from utils.pytorch_model_kit import *
from utils import init_console_and_file_log
from torch import torch, nn
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import StepLR
import optuna


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def global_model_trial(trial: optuna.Trial, if_trial=True):
    # 解析各种传入的参数
    # mod_parser.add_argument('--weight_decay', type=float, default=1e-4)
    _mod_args = mod_parser.parse_args()
    _path_args = train_parser.parse_args(['path'])
    _oth_args = train_parser.parse_args(['other'])
    _dl_args = train_parser.parse_args(['dl'])

    _path_args.prefix = f'{trial.number}_global'  # 命名模型
    path_dict = get_path_params_dict(_path_args)  # 获取日志文件储存路径
    my_logger = init_console_and_file_log(_path_args.prefix, path_dict['logger_path'])  # 建立logger
    save_params(_mod_args, os.path.join(_path_args.dir_name, f'{_path_args.prefix}.csv'))  # 保存模型训练用参数

    # 在这里调整需要变更的参数
    if if_trial:
        _dl_args.batch_size = trial.suggest_categorical('BATCH_SIZE', [256, 512, 1024])
        _mod_args.pos_weight = trial.suggest_int('POS_WEIGHT', 20, 30)
        _mod_args.neg_weight = trial.suggest_int('NEG_WEIGHT', 5, 10)
        _mod_args.learning_rate = trial.suggest_float("LR", 1e-4, 1e-2)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2)
        step_size = trial.suggest_int('step_size', 5, 10)
        gamma = trial.suggest_float('gamma', 0.1, 0.8, step=0.1)
        momentum = trial.suggest_float('momentum', 0.05, 0.2, step=0.01)
    else:
        momentum = 0.08
        weight_decay = 1e-4
        step_size = 5
        gamma = 0.8

    data_loader_dict = get_data_loader(_dl_args)
    model = GlobalTaskDnn()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=_mod_args.learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    my_trainer = PyTorchTrainer(model, nn.BCELoss(), optimizer, scheduler,
                                data_loader_dict['train'], data_loader_dict['val'], data_loader_dict['test'],
                                _mod_args.epochs,
                                path_dict['process_csv_path'], path_dict['test_csv_path'], path_dict['image_path'],
                                path_dict['model_file_path'], path_dict['prc_path'],
                                my_logger, _mod_args.neg_weight, _mod_args.pos_weight, True, trial)
    return my_trainer.train()


def logistic_model(trial: optuna.Trial, if_trial=True):
    _mod_args = mod_parser.parse_args()
    _path_args = train_parser.parse_args(['path'])
    _oth_args = train_parser.parse_args(['other'])
    _dl_args = train_parser.parse_args(['dl'])

    _path_args.prefix = f'{trial.number}_logistic'  # 命名模型
    path_dict = get_path_params_dict(_path_args)  # 获取日志文件储存路径
    my_logger = init_console_and_file_log(_path_args.prefix, path_dict['logger_path'])  # 建立logger
    save_params(_mod_args, os.path.join(_path_args.dir_name, f'{_path_args.prefix}.csv'))  # 保存模型训练用参数

    data_loader_dict = get_data_loader(_dl_args)
    model = LogisticRegression(151, 1)  # 创建一个网络
    model.initialize_weights()  # 初始化权值
    model.to(device)

    # 在这里调整需要变更的参数
    if if_trial:
        _dl_args.batch_size = trial.suggest_categorical('BATCH_SIZE', [256, 512, 1024])
        _mod_args.pos_weight = trial.suggest_int('POS_WEIGHT', 20, 30)
        _mod_args.neg_weight = trial.suggest_int('NEG_WEIGHT', 5, 10)
        _mod_args.learning_rate = trial.suggest_float("LR", 1e-4, 1e-2)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2)
        step_size = trial.suggest_int('step_size', 5, 10)
        gamma = trial.suggest_float('gamma', 0.1, 0.8, step=0.1)
        momentum = trial.suggest_float('momentum', 0.05, 0.2, step=0.01)
    else:
        momentum = 0.08
        weight_decay = 1e-4
        step_size = 5
        gamma = 0.8

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=_mod_args.learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    my_trainer = PyTorchTrainer(model, nn.BCELoss(), optimizer, scheduler,
                                data_loader_dict['train'], data_loader_dict['val'], data_loader_dict['test'],
                                _mod_args.epochs,
                                path_dict['process_csv_path'], path_dict['test_csv_path'], path_dict['image_path'],
                                path_dict['model_file_path'], path_dict['prc_path'],
                                my_logger, _mod_args.neg_weight, _mod_args.pos_weight, True, trial)
    return my_trainer.train()


def cnn_1d_model(trial: optuna.Trial, if_trial=True):
    _mod_args = mod_parser.parse_args()
    _path_args = train_parser.parse_args(['path'])
    _oth_args = train_parser.parse_args(['other'])
    _dl_args = train_parser.parse_args(['dl'])

    _path_args.prefix = f'{trial.number}_1d_cnn'  # 命名模型
    path_dict = get_path_params_dict(_path_args)  # 获取日志文件储存路径
    my_logger = init_console_and_file_log(_path_args.prefix, path_dict['logger_path'])  # 建立logger
    save_params(_mod_args, os.path.join(_path_args.dir_name, f'{_path_args.prefix}.csv'))  # 保存模型训练用参数

    data_loader_dict = get_data_loader(_dl_args)
    model = MyCNN1D()
    model.to(device)

    # 在这里调整需要变更的参数
    if if_trial:
        _mod_args.pos_weight = trial.suggest_int('POS_WEIGHT', 20, 30)
        _mod_args.neg_weight = trial.suggest_int('NEG_WEIGHT', 5, 10)
        _mod_args.learning_rate = trial.suggest_float("LR", 1e-4, 1e-2)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2)
        step_size = trial.suggest_int('step_size', 5, 10)
        gamma = trial.suggest_float('gamma', 0.1, 0.8, step=0.1)
        momentum = trial.suggest_float('momentum', 0.05, 0.2, step=0.01)
    else:
        momentum = 0.08
        weight_decay = 1e-4
        step_size = 5
        gamma = 0.8

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=_mod_args.learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    my_trainer = PyTorchTrainer(model, nn.BCELoss(), optimizer, scheduler,
                                data_loader_dict['train'], data_loader_dict['val'], data_loader_dict['test'],
                                _mod_args.epochs,
                                path_dict['process_csv_path'], path_dict['test_csv_path'], path_dict['image_path'],
                                path_dict['model_file_path'], path_dict['prc_path'],
                                my_logger, _mod_args.neg_weight, _mod_args.pos_weight, True, trial)
    return my_trainer.train()


def saint_model(trial: optuna.Trial):
    _mod_args = mod_parser.parse_args()
    _path_args = train_parser.parse_args(['path'])
    _oth_args = train_parser.parse_args(['other'])
    _dl_args = train_parser.parse_args(['dl'])

    _path_args.prefix = f'{trial.number}_saint'  # 命名模型
    path_dict = get_path_params_dict(_path_args)  # 获取日志文件储存路径
    my_logger = init_console_and_file_log(_path_args.prefix, path_dict['logger_path'])  # 建立logger
    save_params(_mod_args, os.path.join(_path_args.dir_name, f'{_path_args.prefix}.csv'))  # 保存模型训练用参数

    # 读取数据集，获取数据集特征
    df = pandas.read_csv('myds.csv', index_col=0)
    df.reset_index(inplace=True, drop=True)
    label = df['151']
    data_df = df.drop(columns=['151'])

    continuous_cols = [str(i) for i in range(0, 32)]
    cat_dims = [len(df.iloc[:, i].unique()) for i in range(32, 151)]
    cat_embed_input = [(str(u), i) for u, i in zip(range(32, 151), cat_dims)]
    colnames = [str(i) for i in range(151)]
    column_idx = {k: v for v, k in enumerate(colnames)}

    # 构建数据集
    data_set = CustomDataset(data_df, label)
    train_len = int(round(len(data_set) * 0.8, 0))
    test_len = int(round(len(data_set) * 0.1, 0))
    train_len += len(data_set) - train_len - test_len * 2
    train_dataset, val_dataset, test_dataset = random_split(
        dataset=data_set, lengths=[train_len, test_len, test_len], generator=torch.Generator().manual_seed(0)
    )

    data_set_dict = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    _data_loader_dict = {k: DataLoader(data_set_dict[k],
                                       shuffle=True,
                                       batch_size=_dl_args.batch_size,
                                       num_workers=_dl_args.num_work,
                                       pin_memory=_dl_args.if_pin_memory) for k in data_set_dict.keys()}

    # 在这里调整需要变更的参数，第一部分是常规的训练参数
    _mod_args.pos_weight = trial.suggest_int('POS_WEIGHT', 20, 30)
    _mod_args.neg_weight = trial.suggest_int('NEG_WEIGHT', 5, 10)
    _mod_args.learning_rate = trial.suggest_float("LR", 1e-4, 1e-2)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2)
    step_size = trial.suggest_int('step_size', 5, 10)
    gamma = trial.suggest_float('gamma', 0.1, 0.8, step=0.1)
    momentum = trial.suggest_float('momentum', 0.05, 0.2, step=0.01)

    # 第二部分是模型的独特参数
    cat_embed_dropout = trial.suggest_float("cat_embed_dropout", 0.01, 0.3)
    shared_embed = trial.suggest_categorical('shared_embed', [True, False])
    n_heads = trial.suggest_int('n_heads', 6, 8)
    input_dim = n_heads * 2
    n_blocks = trial.suggest_int('n_blocks', 1, 3)
    ff_dropout = trial.suggest_float('ff_dropout', 0.05, 0.2, step=0.01)
    mlp_dropout = trial.suggest_float('mlp_dropout', 0.05, 0.2, step=0.01)
    mlp_linear_first = trial.suggest_categorical('mlp_linear_first', [True, False])
    mlp_type = trial.suggest_categorical('mlp_type', [1, 2])
    mlp_type_dict = {1: [64, 1], 2: [128, 64, 1]}

    try:
        model = nn.Sequential(SAINT(column_idx=column_idx,
                                    cat_embed_input=cat_embed_input,
                                    continuous_cols=continuous_cols,
                                    mlp_hidden_dims=mlp_type_dict[mlp_type],
                                    cat_embed_dropout=cat_embed_dropout,
                                    shared_embed=shared_embed,
                                    n_heads=n_heads,
                                    n_blocks=n_blocks,
                                    ff_dropout=ff_dropout,
                                    mlp_dropout=mlp_dropout,
                                    mlp_linear_first=mlp_linear_first,
                                    input_dim=input_dim), nn.Sigmoid())
        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=_mod_args.learning_rate,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
        scheduler = StepLR(optimizer,
                           step_size=step_size,
                           gamma=gamma)

        my_trainer = PyTorchTrainer(model, nn.BCELoss(), optimizer, scheduler,
                                    _data_loader_dict['train'], _data_loader_dict['val'], _data_loader_dict['test'],
                                    _mod_args.epochs,
                                    path_dict['process_csv_path'], path_dict['test_csv_path'], path_dict['image_path'],
                                    path_dict['model_file_path'], path_dict['prc_path'],
                                    my_logger, _mod_args.neg_weight, _mod_args.pos_weight, True, trial)
        value = my_trainer.train()
        gc.collect()
        torch.cuda.empty_cache()
        return value
    except:
        my_logger.error(traceback.format_exc())
        return None


def tab_transform_model(trial: optuna.Trial):
    _mod_args = mod_parser.parse_args()
    _path_args = train_parser.parse_args(['path'])
    _oth_args = train_parser.parse_args(['other'])
    _dl_args = train_parser.parse_args(['dl'])

    _path_args.prefix = f'{trial.number}_tab_trs'  # 命名模型
    path_dict = get_path_params_dict(_path_args)  # 获取日志文件储存路径
    my_logger = init_console_and_file_log(_path_args.prefix, path_dict['logger_path'])  # 建立logger
    save_params(_mod_args, os.path.join(_path_args.dir_name, f'{_path_args.prefix}.csv'))  # 保存模型训练用参数

    # 读取数据集，获取数据集特征
    df = pandas.read_csv('myds.csv', index_col=0)
    df.reset_index(inplace=True, drop=True)
    label = df['151']
    data_df = df.drop(columns=['151'])

    continuous_cols = [str(i) for i in range(0, 32)]
    cat_dims = [len(df.iloc[:, i].unique()) for i in range(32, 151)]
    cat_embed_input = [(str(u), i) for u, i in zip(range(32, 151), cat_dims)]
    colnames = [str(i) for i in range(151)]
    column_idx = {k: v for v, k in enumerate(colnames)}

    # 构建数据集
    data_set = CustomDataset(data_df, label)
    train_len = int(round(len(data_set) * 0.8, 0))
    test_len = int(round(len(data_set) * 0.1, 0))
    train_len += len(data_set) - train_len - test_len * 2
    train_dataset, val_dataset, test_dataset = random_split(
        dataset=data_set, lengths=[train_len, test_len, test_len], generator=torch.Generator().manual_seed(0)
    )

    data_set_dict = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    _data_loader_dict = {k: DataLoader(data_set_dict[k],
                                       shuffle=True,
                                       batch_size=_dl_args.batch_size,
                                       num_workers=_dl_args.num_work,
                                       pin_memory=_dl_args.if_pin_memory) for k in data_set_dict.keys()}

    # 在这里调整需要变更的参数，第一部分是常规的训练参数
    _mod_args.pos_weight = trial.suggest_int('POS_WEIGHT', 20, 30)
    _mod_args.neg_weight = trial.suggest_int('NEG_WEIGHT', 5, 10)
    _mod_args.learning_rate = trial.suggest_float("LR", 1e-4, 1e-2)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2)
    step_size = trial.suggest_int('step_size', 5, 10)
    gamma = trial.suggest_float('gamma', 0.1, 0.8, step=0.1)
    momentum = trial.suggest_float('momentum', 0.05, 0.2, step=0.01)

    # 第二部分是模型的独特参数
    cat_embed_dropout = trial.suggest_float("cat_embed_dropout", 0.01, 0.3)
    shared_embed = trial.suggest_categorical('shared_embed', [True, False])
    n_heads = trial.suggest_int('n_heads', 6, 8)
    input_dim = n_heads * 2
    n_blocks = trial.suggest_int('n_blocks', 1, 3)
    ff_dropout = trial.suggest_float('ff_dropout', 0.05, 0.2, step=0.01)
    mlp_dropout = trial.suggest_float('mlp_dropout', 0.05, 0.2, step=0.01)
    mlp_linear_first = trial.suggest_categorical('mlp_linear_first', [True, False])
    mlp_type = trial.suggest_categorical('mlp_type', [1, 2])
    mlp_type_dict = {1: [64, 1], 2: [128, 64, 1]}

    try:
        model = nn.Sequential(TabTransformer(column_idx=column_idx,
                                             cat_embed_input=cat_embed_input,
                                             continuous_cols=continuous_cols,
                                             mlp_hidden_dims=mlp_type_dict[mlp_type],
                                             cat_embed_dropout=cat_embed_dropout,
                                             shared_embed=shared_embed,
                                             n_heads=n_heads,
                                             n_blocks=n_blocks,
                                             ff_dropout=ff_dropout,
                                             mlp_dropout=mlp_dropout,
                                             mlp_linear_first=mlp_linear_first,
                                             input_dim=input_dim), nn.Sigmoid())
        model.to(device)

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=_mod_args.learning_rate,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
        scheduler = StepLR(optimizer,
                           step_size=step_size,
                           gamma=gamma)

        my_trainer = PyTorchTrainer(model, nn.BCELoss(), optimizer, scheduler,
                                    _data_loader_dict['train'], _data_loader_dict['val'], _data_loader_dict['test'],
                                    _mod_args.epochs,
                                    path_dict['process_csv_path'], path_dict['test_csv_path'], path_dict['image_path'],
                                    path_dict['model_file_path'], path_dict['prc_path'],
                                    my_logger, _mod_args.neg_weight, _mod_args.pos_weight, True, trial)
        value = my_trainer.train()
        gc.collect()
        torch.cuda.empty_cache()
        return value
    except:
        my_logger.error(traceback.format_exc())
        return None


def trial_task(study_name, study_func):
    study = optuna.create_study(
        storage=f"sqlite:///{study_name}.sqlite3",  # 数据文件存放的地址
        study_name=study_name,  # 需要指定学习任务的名字，该名字就是数据文件的名字
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=30),
        load_if_exists=True)

    study.optimize(study_func, n_trials=70)
    print(f'best params:{study.best_params}, best value:{study.best_value}, best trial:{study.best_trial}')


if __name__ == '__main__':
    trial_task('saint', saint_model)
