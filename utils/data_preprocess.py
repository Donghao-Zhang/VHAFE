from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from envs.afe.evaluater import Evaluater
from scipy.io.arff import loadarff
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import copy
import math

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore


def percentna(dataframe, threshold):
    columns = dataframe.columns[(dataframe.isna().sum()/dataframe.shape[0])>threshold]
    return columns.tolist()


# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


def corrwith_target(dataframe, target, threshold):
    cor = dataframe.corr()
    # Correlation with output variable
    cor_target = abs(cor[target])
    # Selecting non correlated features
    relevant_features = cor_target[cor_target<threshold]
    return relevant_features.index.tolist()[:-1]


def load_test_data(args):
    features = np.array(np.arange(30)).reshape([6, 5])
    targets = np.array([0, 1, 3, 6, 4, 8]).reshape([6, 1])

    args.greater_is_better = False
    args.tasktype = "R"
    # args.evaluatertype = 'rf'

    # tasktype 任务类型（R代表回归），evaluatertype代表效果测试模型（rf代表随机森林）
    evaluater = Evaluater(cv=5, random_state=np.random.randint(100000), tasktype=args.tasktype, evaluatertype=args.evaluatertype,
                          n_jobs=8, greater_is_better=args.greater_is_better)

    print('initial rmse is: {:.4f}'.format(evaluater.CV2(features, targets)))
    return features, targets, evaluater, args


def load_arff_data(f_path, args):
    print(f"load dataset from: {f_path}")
    dataset, meta = loadarff(f_path)
    dataset = np.array(dataset.tolist())
    features = dataset[:, :-1]
    targets = dataset[:, -1][:, None]

    # 参数设置
    args.greater_is_better = True
    args.tasktype = "R"
    # args.evaluatertype = 'rf'

    evaluater = Evaluater(cv=5, random_state=np.random.randint(100000), tasktype=args.tasktype, evaluatertype='rf',
                          n_jobs=8, greater_is_better=args.greater_is_better, default_param=True)
    print('initial 1-RAE is: {:.4f}'.format(evaluater.CV2(features, targets)))
    return features, targets, evaluater, args


def load_uci_data(f_dir, data_name, args):
    print(f"load dataset from: {f_dir}, load data: {data_name}")
    if data_name in ['housing', 'airfoil']:
        args.tasktype = "R"
        if data_name == 'housing':
            data_name = 'housing.data'
        else:
            data_name = 'airfoil.dat'
        data_path = os.path.join(f_dir, 'r', data_name)
        dataset = np.loadtxt(data_path)
        features = dataset[:, :-1]
        targets = dataset[:, -1][:, None]
    else:
        args.tasktype = "C"
        delimiter = None
        target_pos = -1
        read_use = 'np'
        header = None
        index_col = None
        if data_name == 'german':  # 样本数目与论文中不一致，多1个
            data_name = 'german.data'
        elif data_name == 'pimaindian':
            data_name = 'PimaIndian.arff'
            read_use = 'arff'
        elif data_name == 'spectf':
            data_name = 'SPECTF.data'
            delimiter = ','
            target_pos = 0
        elif data_name == 'ionosphere':
            data_name = 'ionosphere.data'
            delimiter = ','
            read_use = 'csv'
        elif data_name == 'credit_default':  # 特征数目不一致
            data_name = 'credit_default.xls'
            read_use = 'excel'
            header = [0, 1]
            index_col = 0
        elif data_name == 'messidor_features':  # 样本数目与论文中不一致，多1个
            data_name = 'messidor_features.arff'
            read_use = 'arff'
        elif data_name == 'spambase':
            data_name = 'spambase.data'
            delimiter = ','
        elif data_name == 'fertility':
            data_name = 'fertility.txt'
            read_use = 'csv'
            delimiter = ','
        elif data_name == 'megawatt1':
            data_name = 'MegaWatt1.arff'
            read_use = 'arff'

        data_path = os.path.join(f_dir, 'c', data_name)
        if read_use == 'np':
            dataset = np.loadtxt(data_path, delimiter=delimiter)
        elif read_use == 'csv':
            dataset = pd.read_csv(data_path, delimiter=delimiter, header=header, index_col=index_col)
            dataset = dataset.values
        elif read_use == 'excel':
            dataset = pd.read_excel(data_path, header=header, index_col=index_col)
            dataset = dataset.values
        elif read_use == 'arff':
            dataset, meta = loadarff(data_path)
            dataset = np.array(dataset.tolist())
        else:
            raise ValueError
        if target_pos < 0:
            features = dataset[:, :target_pos]
        else:
            features = np.concatenate([dataset[:, :target_pos], dataset[:, target_pos+1:]], axis=1)
        features = features.astype(np.float_)
        targets = dataset[:, target_pos]
        le = preprocessing.LabelEncoder()
        targets = le.fit_transform(targets)[:, None]
    

    # 参数设置
    args.greater_is_better = True
    # args.evaluatertype = 'rf'
    evaluater = Evaluater(cv=5, random_state=np.random.randint(100000), tasktype=args.tasktype, evaluatertype=args.evaluatertype,
                          n_jobs=8, greater_is_better=args.greater_is_better, default_param=True)
    if args.tasktype == "R":
        print('initial 1-RAE is: {:.4f}'.format(evaluater.CV2(features, targets)))
    else:
        print('initial F1 is: {:.4f}'.format(evaluater.CV2(features, targets)))
    return features, targets, evaluater, args


def load_kaggle_data(f_path, args):  # 0.9918
    print(f"load dataset from: {f_path}")

    dataset = pd.read_csv(f_path, header=0)
    # dataset['datetime']

    dataset = dataset.iloc[:, 1:].values
    features = dataset[:, :-1]
    targets = dataset[:, -1][:, None]

    # 参数设置
    args.greater_is_better = True
    args.tasktype = "R"
    # args.evaluatertype = 'rf'

    evaluater = Evaluater(cv=5, random_state=np.random.randint(100000), tasktype=args.tasktype, evaluatertype='rf',
                          n_jobs=8, greater_is_better=args.greater_is_better, default_param=True)
    print('initial 1-RAE is: {:.4f}'.format(evaluater.CV2(features, targets)))
    return features, targets, evaluater, args


def load_SECOM(f_path, args, random_state=1):
    x = pd.read_csv(os.path.join(f_path, 'secom.data'), sep=' ', header=None)
    y = pd.read_csv(os.path.join(f_path, 'secom_labels.data'), sep=' ', header=None).iloc[:, 0]

    original_feat_dim = x.shape[1]
    # data = pd.read_csv(f_path, sep=',')
    # x = data.drop('Pass/Fail', axis=1).iloc[:, 1:]
    # y = data['Pass/Fail']

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    # 1. 去除缺失值过多的特征
    na_columns = percentna(x, 0.5)
    x = x.drop(na_columns, axis=1)

    # 2. 缺失值填充
    # imputer = KNNImputer()
    # x = imputer.fit_transform(x)

    # 3. 标准化
    x = StandardScaler().fit_transform(x)
    x = pd.DataFrame(x).fillna(0)

    # 3. 去除方差小于一定阈值的特征
    selector = VarianceThreshold(threshold=0)
    x = selector.fit_transform(x)

    # 4. 去除与其他特征强相关的特征
    x = pd.DataFrame(x)
    corr_features = correlation(x, 0.95)
    x = x.drop(corr_features, axis=1)

    # 5. 选择与目标强相关的特征
    dummy_train = x.copy()
    dummy_train['target'] = y
    corrwith_cols = corrwith_target(dummy_train, 'target', 0.05)
    x = x.drop(corrwith_cols, axis=1)

    # x_train, y_train = SMOTE().fit_resample(x, y_train)

    x = x.to_numpy()

    # 参数设置
    args.greater_is_better = True
    args.tasktype = "C"
    # args.evaluatertype = 'lr'

    evaluater = Evaluater(cv=5, random_state=np.random.randint(100000), tasktype=args.tasktype, evaluatertype=args.evaluatertype,
                          n_jobs=8, greater_is_better=args.greater_is_better, use_cuml=False)

    # 6. RFECV Recursive feature elimination with cross-validation
    # if args.evaluatertype == 'rf':
    clf = RandomForestClassifier(n_estimators=20, random_state=random_state)
    # elif args.evaluatertype == "lr":
    #     clf = LogisticRegression(solver='liblinear', C=200, dual=False, random_state=random_state)
    # else:
    #     raise ValueError
    rfecv = RFECV(
        estimator=clf,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        scoring='f1')
    x = rfecv.fit_transform(x, y[:, None])

    print(f'original number of features is {original_feat_dim}, there are {x.shape[1]} features left.')

    print('initial f1 score is: {:.4f}'.format(evaluater.CV2(x, y[:, None])))
    return x, y[:, None], evaluater, args


def extract_physical_input(f_path, samples):
    indices = samples.columns.tolist()
    tag_name2unit = {}
    for index in indices:
        if (index[0], index[1]) not in tag_name2unit.keys():
            tag_name2unit[(index[0], index[1])] = index[2]
        else:
            raise ValueError

    path = os.path.join(f_path, "某燃机现场数据名称翻译与故障变量关联v2.xlsx")
    input_variable = pd.read_excel(path, header=0)
    tags = []
    names = []
    units = []

    for i in range(input_variable.shape[0]):
        item = input_variable.iloc[i]
        if item["变量类型"] == "物理输入" or (
                item["Tag"] == "<08>CEA.0099" and
                item["Name"] == "GEN. OUTPUT (ACTLD)" and item["Unit"] == "MW"
        ):
            tags.append(item["Tag"])
            names.append(item["Name"])
            if pd.isna(item["Unit"]):
                units.append(tag_name2unit[(item["Tag"], item["Name"])])
            else:
                units.append(item["Unit"])
    label_index = pd.MultiIndex.from_arrays([tags, names, units])
    sample_timestamps = samples[samples.columns[0]]
    samples = samples[label_index]
    return samples, sample_timestamps


def preprocess_turbine_data(samples, sample_timestamps):
    label_index = pd.MultiIndex.from_arrays([["<08>CEA.0099"], ["GEN. OUTPUT (ACTLD)"], ["MW"]])
    indices = samples.columns.tolist()
    label_position = [(i, index) for i, index in enumerate(indices) if index == label_index.tolist()[0]][0]
    labels = samples[label_index].values
    samples = samples.values
    features = np.concatenate([samples[:, :label_position[0]], samples[:, label_position[0]+1:]], axis=-1)
    indices = indices[:label_position[0]] + indices[label_position[0]+1:] + [indices[label_position[0]]]
    sample_ids = np.where(labels > 1)[0]
    sample_timestamps = sample_timestamps.iloc[sample_ids]
    features = features[sample_ids]
    labels = labels[sample_ids]
    samples = np.concatenate([features, labels], axis=-1)
    samples = samples.astype(float)
    assert samples.shape[1] == len(indices)
    return samples, indices, sample_timestamps


def load_turbine_data(f_path, args, num_dataset=5, random_state=1):
    samples = []
    for i in range(1, num_dataset + 1):
        path = os.path.join(f_path, "G1M1W%d.csv" % i)
        samples.append(pd.read_csv(path, index_col=None, header=[0, 1, 2]))
    # path = os.path.join(f_path, "G1M1W%d.csv" % 5)
    samples.append(pd.read_csv(path, index_col=None, header=[0, 1, 2]))
    samples = pd.concat(samples, axis=0)
    samples, sample_timestamps = extract_physical_input(f_path, samples)
    samples, indices, sample_timestamps = preprocess_turbine_data(samples, sample_timestamps)

    feature_names = []
    for f_n in indices:
        curr_f = ""
        if f_n[0].strip():
            curr_f += f_n[0].strip()
        if f_n[1].strip():
            curr_f += "/" + f_n[1].strip()
        if f_n[2].strip():
            curr_f += "/" + f_n[2].strip()
        feature_names.append(curr_f)

    x, y = samples[:, :-1], samples[:, -1]

    original_feat_dim = x.shape[1]

    # 3. 标准化
    x = StandardScaler().fit_transform(x)

    # 3. 去除方差小于一定阈值的特征
    x = pd.DataFrame(x, columns=feature_names[:-1])
    selector = VarianceThreshold(threshold=0)
    x = selector.fit_transform(x)
    out_features_names = selector.get_feature_names_out()

    # 4. 去除与其他特征强相关的特征
    x = pd.DataFrame(x, columns=out_features_names)
    corr_features = correlation(x, 0.95)
    x = x.drop(corr_features, axis=1)

    # 5. 选择与目标强相关的特征
    dummy_train = x.copy()
    dummy_train['target'] = y
    corrwith_cols = corrwith_target(dummy_train, 'target', 0.1)
    x = x.drop(corrwith_cols, axis=1)

    out_features_names = x.columns
    x = x.to_numpy()

    # 参数设置
    args.greater_is_better = True
    args.tasktype = "R"
    # args.evaluatertype = 'rf'

    # evaluater = []
    random_state = np.random.randint(100000)
    evaluater = Evaluater(cv=5, random_state=random_state, tasktype=args.tasktype, evaluatertype=args.evaluatertype,
                          n_jobs=8, greater_is_better=args.greater_is_better, use_cuml=False, max_depth=8, n_bins=16)
    evaluater = [
        copy.deepcopy(evaluater) for _ in range(args.n_rollout_threads)
    ]
    # for i in range(args.n_rollout_threads):
    #     evaluater.append(
    #         Evaluater(cv=5, random_state=random_state, tasktype=args.tasktype, evaluatertype=args.evaluatertype,
    #                   n_jobs=8, greater_is_better=args.greater_is_better, use_cuml=False, max_depth=8, n_bins=16))

    print(f'original number of features is {original_feat_dim}, there are {x.shape[1]} features left.')
    # cuml.ensemble.RandomForestRegressor 即使设置random_state一致，目前也还无法保证完全一致的结果
    # rmse = None
    # for i in range(len(evaluater)):
    #     if rmse is not None:
    #         assert rmse == evaluater[i].CV2(x, y[:, None])
    #     else:
    #         rmse = evaluater[i].CV2(x, y[:, None])
    print('initial rmse score is: {:.4f}'.format(evaluater[0].CV2(x, y[:, None])))
    print(evaluater[0].CV2_test(x, y[:, None]))
    return x, y[:, None], evaluater, args


def load_simulation_data_flat(args, classification=False):
    n_samples = 5000
    n_features = 20
    features = np.random.normal(loc=0, scale=1, size=(n_samples, n_features))

    mmn = MinMaxScaler(feature_range=(0, 1))
    # mmn + sqrt
    y = mmn.fit_transform(features[:, 0][:, np.newaxis]).flatten()
    targets = np.sqrt(y)
    # zscore
    targets += zscore(features[:, 1])
    # sigmoid
    targets += (1 + getattr(np, 'tanh')(features[:, 2] / 2)) / 2
    # tanh
    targets += getattr(np, 'tanh')(features[:, 3])
    # round
    targets += getattr(np, 'round')(features[:, 4])
    # square
    targets += getattr(np, 'square')(features[:, 5])
    # log
    vmin = features[:, 6].min()
    targets += np.log(features[:, 6] - vmin + 1) if vmin < 1 else np.log(features[:, 6])
    # reciprocal
    new_feature = 1 / features[:, 7]
    new_feature[features[:, 7] == 0] = 0
    targets += new_feature
    # minus
    targets += features[:, 8] - features[:, 9]
    # sum
    targets += features[:, 10] + features[:, 11]
    # div
    f1 = features[:, 12]
    f2 = features[:, 13]
    y = f1 / f2
    y[f2 == 0] = 0
    mmn = MinMaxScaler()
    targets += mmn.fit_transform(y[:, np.newaxis]).flatten()
    # time
    y = features[:, 14] * features[:, 15]
    mmn = MinMaxScaler()
    targets += mmn.fit_transform(y[:, np.newaxis]).flatten()

    y = np.mod(features[:, 16], features[:, 17])
    y[features[:, 17] == 0] = 0
    targets += y

    targets = targets[:, None]

    if classification:
        mmn = MinMaxScaler(feature_range=(-1, 1))
        targets = mmn.fit_transform(targets)
        targets[np.where(targets > 0)] = 1
        targets[np.where(targets <= 0)] = 0
        args.greater_is_better = True
        args.tasktype = "C"
        # args.evaluatertype = 'rf'
    else:
        args.greater_is_better = True
        args.tasktype = "R"
        # args.evaluatertype = 'rf'

    # tasktype 任务类型（R代表回归），evaluatertype代表效果测试模型（rf代表随机森林）
    evaluater = Evaluater(cv=5, random_state=np.random.randint(100000), tasktype=args.tasktype, evaluatertype=args.evaluatertype,
                          n_jobs=8, greater_is_better=args.greater_is_better)

    if classification:
        print('initial f1 is: {:.4f}'.format(evaluater.CV2(features, targets)))
    else:
        print('initial 1-RAE is: {:.4f}'.format(evaluater.CV2(features, targets)))
    return features, targets, evaluater, args


def load_simulation_data_random(args, actions, use_o2=True, classification=False):
    n_samples = 5000
    n_features = 10
    n_empty_features = 2
    episode_length = 20
    features = np.random.normal(loc=0, scale=1, size=(n_samples, n_features))

    # 图初始化
    graphs = []
    for i in range(n_features):
        node_state = {'visibility': True, 'feature': features[:, i], 'parent': -1}
        graphs.append({0: node_state})
    curr_node_positions = [0] * n_features
    dones = [False] * n_features

    action_num = args.o2_start if not use_o2 else len(actions)

    curr_episode = 0
    while curr_episode < episode_length and not np.all(dones):
        curr_episode += 1
        act = np.random.choice(range(action_num))  # 选择动作
        act_str = actions[act]
        f_id = np.random.choice(range(n_features))  # 选择特征
        f_id_2 = np.random.choice(range(n_features))  # 选择o2特征
        curr_node_id = curr_node_positions[f_id]
        curr_feature = graphs[f_id][curr_node_id]['feature']
        if dones[f_id]:
            curr_episode -= 1
            continue
        if act_str in {'</s>', 'hidden', 'back', 'unchanged'}:
            if act_str == '</s>':
                dones[f_id] = True
            elif act_str == 'hidden':
                graphs[f_id][curr_node_id]['visibility'] = False
            elif act_str == 'back':
                if graphs[f_id][curr_node_id]['parent'] != -1:
                    curr_node_positions[f_id] = graphs[f_id][curr_node_id]['parent']
        else:
            next_node_state = {'visibility': True, 'feature': None, 'parent': curr_node_id}
            if act_str in {'square', 'tanh', 'round'}:
                new_feature = getattr(np, act_str)(curr_feature)
            elif act_str == "log":
                vmin = curr_feature.min()
                new_feature = np.log(curr_feature - vmin + 1) if vmin < 1 else np.log(curr_feature)
            elif act_str == "sqrt":
                vmin = curr_feature.min()
                new_feature = np.sqrt(curr_feature - vmin) if vmin < 0 else np.sqrt(curr_feature)
            elif act_str == "mmn":
                mmn = MinMaxScaler()
                new_feature = mmn.fit_transform(curr_feature[:, np.newaxis]).flatten()
            elif act_str == "sigmoid":
                new_feature = (1 + getattr(np, 'tanh')(curr_feature / 2)) / 2
            elif act_str == "zscore":
                if np.var(curr_feature) != 0:
                    new_feature = zscore(curr_feature)
                else:
                    new_feature = np.zeros_like(curr_feature)
            elif act_str == 'reciprocal':
                new_feature = 1 / curr_feature
                new_feature[curr_feature == 0] = 0
            # order 2
            elif act_str == 'sum':
                new_feature = curr_feature + graphs[f_id_2][curr_node_positions[f_id_2]]['feature']
            elif act_str == 'minus':
                new_feature = curr_feature - graphs[f_id_2][curr_node_positions[f_id_2]]['feature']
            elif act_str == 'div':
                new_feature = curr_feature / graphs[f_id_2][curr_node_positions[f_id_2]]['feature']
                new_feature[graphs[f_id_2][curr_node_positions[f_id_2]]['feature'] == 0] = 0
                new_feature = np.clip(new_feature, -math.sqrt(3.4e38), math.sqrt(3.4e38))
                mmn = MinMaxScaler()
                new_feature = mmn.fit_transform(new_feature[:, np.newaxis]).flatten()
            elif act_str == 'time':
                new_feature = curr_feature * graphs[f_id_2][curr_node_positions[f_id_2]]['feature']
                new_feature = np.clip(new_feature, -math.sqrt(3.4e38), math.sqrt(3.4e38))
                mmn = MinMaxScaler()
                new_feature = mmn.fit_transform(new_feature[:, np.newaxis]).flatten()
            elif act_str == 'modulo':
                new_feature = np.mod(curr_feature, graphs[f_id_2][curr_node_positions[f_id_2]]['feature'])
                new_feature[graphs[f_id_2][curr_node_positions[f_id_2]]['feature'] == 0] = 0
            else:
                raise ValueError

            new_feature = np.nan_to_num(new_feature)
            new_feature = np.clip(new_feature, -math.sqrt(3.4e38), math.sqrt(3.4e38))
            next_node_state['feature'] = new_feature
            next_node_id = len(graphs[f_id])
            graphs[f_id][next_node_id] = next_node_state
            curr_node_positions[f_id] = next_node_id

    features = np.concatenate([features, np.random.normal(loc=0, scale=1, size=(n_samples, n_empty_features))], axis=1)
    # 生成 targets
    curr_features = np.empty((features.shape[0], 0))
    for graph_states in graphs:
        for node_id in range(len(graph_states)):
            node = graph_states[node_id]
            if node['visibility'] and not np.any(np.all(curr_features == node['feature'][:, None], axis=0)) and \
                    np.max(node['feature']) != np.min(node['feature']):
                curr_features = np.concatenate([curr_features, node['feature'][:, None]], axis=1)
    targets = np.sum(curr_features, axis=1)[:, None]

    if classification:
        mmn = MinMaxScaler(feature_range=(-1, 1))
        targets = mmn.fit_transform(targets)
        targets[np.where(targets > 0)] = 1
        targets[np.where(targets <= 0)] = 0
        args.greater_is_better = True
        args.tasktype = "C"
        # args.evaluatertype = 'rf'
    else:
        args.greater_is_better = True
        args.tasktype = "R"
        # args.evaluatertype = 'rf'

    # tasktype 任务类型（R代表回归），evaluatertype代表效果测试模型（rf代表随机森林）
    evaluater = Evaluater(cv=5, random_state=np.random.randint(100000), tasktype=args.tasktype, evaluatertype=args.evaluatertype,
                          n_jobs=8, greater_is_better=args.greater_is_better)

    if classification:
        print('initial f1 is: {:.4f}'.format(evaluater.CV2(features, targets)))
    else:
        print('initial 1-RAE is: {:.4f}'.format(evaluater.CV2(features, targets)))
    return features, targets, evaluater, args
