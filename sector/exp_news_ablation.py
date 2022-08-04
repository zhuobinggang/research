from sector import *
from mainichi_paragraph import read_ld_train, read_ld_tests, read_ld_test, read_ld_dev
from novel_learning_curve import RANDOM_SEEDs, save_dic, create_iteration_callback_shell
from exp_novel import create_model_with_seed

dic = {
    'LEFT_AUX0': [],
    'LEFT_AUX1': [],
    'LEFT_AUX2': [],
    'RIGHT_AUX0': [],
    'RIGHT_AUX1': [],
    'RIGHT_AUX2': [],
    'NO_AUX0': [],
    'NO_AUX1': [],
    'NO_AUX2': [],
    'COUTER_AUX0': [],
    'COUTER_AUX1': [],
    'COUTER_AUX2': [],
}

def create_model_with_seed(seed):
    t.manual_seed(seed)
    m = Sector_2022()
    time_string = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'created model with seed {seed} at time {time_string}')
    return m

def ablation_loss_left_aux(m, ss, labels, fl_rate, aux_rate):
    bert = m.bert
    toker = m.toker
    combined_ids, sep_idxs = encode(ss, toker, [True, True, False, False])
    labels = [int(label) if label is not None else None for label in labels] # 处理np random带来的int 变成string的问题
    labels = [labels[1], labels[2]] # (3)，不需要第一个label
    out_bert = bert(combined_ids.unsqueeze(0).cuda()).last_hidden_state[:, sep_idxs, :] # (1, 2, 768)
    out_mlp = m.classifier(out_bert) # (1, 2, 1)
    # cal loss
    loss = focal_aux_loss( # 不使用focal loss & 不使用辅助损失
            out_mlp.squeeze(), 
            labels, 
            fl_rate = fl_rate, 
            aux_rate = aux_rate,
            main_pos = 1) # NOTE: 1
    return loss

def ablation_loss_right_aux(m, ss, labels, fl_rate, aux_rate):
    bert = m.bert
    toker = m.toker
    combined_ids, sep_idxs = encode(ss, toker, [False, True, True, False])
    labels = [int(label) if label is not None else None for label in labels] # 处理np random带来的int 变成string的问题
    labels = [labels[2], labels[3]] # (2)，不需要第一个label
    out_bert = bert(combined_ids.unsqueeze(0).cuda()).last_hidden_state[:, sep_idxs, :] # (1, 2, 768)
    out_mlp = m.classifier(out_bert) # (1, 2, 1)
    # cal loss
    loss = focal_aux_loss( # 不使用focal loss & 不使用辅助损失
            out_mlp.squeeze(), 
            labels, 
            fl_rate = fl_rate, 
            aux_rate = aux_rate,
            main_pos = 0) # NOTE: 0
    return loss

def ablation_loss_no_aux(m, ss, labels, fl_rate, aux_rate = None):
    bert = m.bert
    toker = m.toker
    combined_ids, sep_idxs = encode(ss, toker, [False, True, False, False])
    labels = [int(label) if label is not None else None for label in labels] # 处理np random带来的int 变成string的问题
    label = labels[2] # (1)，不需要第一个label
    out_bert = bert(combined_ids.unsqueeze(0).cuda()).last_hidden_state[:, sep_idxs, :] # (1, 1, 768)
    out_mlp = m.classifier(out_bert) # (1, 1, 1)
    assert out_mlp.shape == (1, 1, 1)
    # cal loss
    loss = focal_loss( # 不使用focal loss & 不使用辅助损失
            out_mlp.squeeze(), # (scalar)
            t.tensor(label).cuda(), # (scalar)
            fl_rate = fl_rate)
    return loss


def ablation_loss_counter_aux(m, ss, labels, fl_rate, aux_rate = None):
    CLS_POS = [0] # baseline需要调用encode_standard（只添加一个sep在中间）然后取出CLS对应的embedding
    bert = m.bert
    toker = m.toker
    combined_ids, _ = encode(ss, toker, [True, False, True, False])
    labels = [int(label) if label is not None else None for label in labels] # 处理np random带来的int 变成string的问题
    label = labels[2] # (1)，不需要第一个label
    out_bert = bert(combined_ids.unsqueeze(0).cuda()).last_hidden_state[:, CLS_POS, :] # (1, 1, 768)
    out_mlp = m.classifier(out_bert) # (1, 1, 1)
    assert out_mlp.shape == (1, 1, 1)
    # cal loss
    loss = focal_loss( # 不使用focal loss & 不使用辅助损失
            out_mlp.squeeze(), # (scalar)
            t.tensor(label).cuda(), # (scalar)
            fl_rate = fl_rate)
    return loss


def train_shell(m, ds_train, loss_function, epoch = 1, batch = 16, fl_rate = 0, aux_rate = 0, iteration_callback = None, random_seed = True):
    first_time = datetime.datetime.now()
    opter = m.opter
    for epoch_idx in range(epoch):
        print(f'Train epoch {epoch_idx}')
        if random_seed:
            np.random.seed(RANDOM_SEED)
        for row_idx, (ss, labels) in enumerate(np.random.permutation(ds_train)):
            if row_idx % 1000 == 0:
                print(f'finished: {row_idx}/{len(ds_train)}')
                pass
            loss = loss_function(m, ss, labels, fl_rate, aux_rate)
            loss.backward()
            # backward
            if (row_idx + 1) % batch == 0:
                if iteration_callback is not None:
                    iteration_callback()
                opter.step()
                opter.zero_grad()
    opter.step()
    opter.zero_grad()
    last_time = datetime.datetime.now()
    delta = last_time - first_time
    print(delta.seconds)
    return delta.seconds

def test_shell(ds_test, m, seps, main_sep_idx_relative):
    y_true = []
    y_pred = []
    toker = m.toker
    bert = m.bert
    for row_idx, row in enumerate(ds_test):
        ss, labels = row
        combined_ids, sep_idxs = encode(ss, toker, seps)
        main_sep_idx = sep_idxs[main_sep_idx_relative] # (1)
        out_bert = bert(combined_ids.unsqueeze(0).cuda()).last_hidden_state[:, main_sep_idx, :] # (1, 768)
        out_mlp = m.classifier(out_bert) # (1, 1)
        assert out_mlp.shape == (1,1)
        y_pred.append(out_mlp.item())
        y_true.append(int(labels[2]))
    return y_true, y_pred


def train_left_aux(m, ds_train, epoch = 1, batch = 16, fl_rate = 0, aux_rate = 0, iteration_callback = None, random_seed = True):
    train_shell(m, ds_train, ablation_loss_left_aux, epoch, batch, fl_rate, aux_rate, iteration_callback, random_seed)


def test_left_aux(m, ds_test):
    return test_shell(ds_test, m, [True, True, False, False], 1)

def train_right_aux(m, ds_train, epoch = 1, batch = 16, fl_rate = 0, aux_rate = 0, iteration_callback = None, random_seed = True):
    train_shell(m, ds_train, ablation_loss_right_aux, epoch, batch, fl_rate, aux_rate, iteration_callback, random_seed)

def test_right_aux(m, ds_test):
    return test_shell(ds_test, m, [False, True, True, False], 0)

def train_no_aux(m, ds_train, epoch = 1, batch = 16, fl_rate = 0, aux_rate = 0, iteration_callback = None, random_seed = True):
    train_shell(m, ds_train, ablation_loss_no_aux, epoch, batch, fl_rate, aux_rate, iteration_callback, random_seed)

def test_no_aux(m, ds_test):
    return test_shell(ds_test, m, [False, True, False, False], 0)

def train_counter_aux(m, ds_train, epoch = 1, batch = 16, fl_rate = 0, aux_rate = 0, iteration_callback = None, random_seed = True):
    train_shell(m, ds_train, ablation_loss_counter_aux, epoch, batch, fl_rate, aux_rate, iteration_callback, random_seed)

def test_counter_aux(m, ds_test):
    y_true = []
    y_pred = []
    toker = m.toker
    bert = m.bert
    CLS_POS = [0]
    for row_idx, row in enumerate(ds_test):
        ss, labels = row
        combined_ids, _ = encode(ss, toker, [True, False, True, False])
        out_bert = bert(combined_ids.unsqueeze(0).cuda()).last_hidden_state[:, CLS_POS, :] # (1, 1, 768)
        out_mlp = m.classifier(out_bert) # (1, 1, 1)
        y_pred.append(out_mlp.squeeze().item())
        y_true.append(int(labels[2]))
    return y_true, y_pred

#     'LEFT_AUX0': [],
#     'LEFT_AUX1': [],
#     'LEFT_AUX2': [],
#     'RIGHT_AUX0': [],
#     'RIGHT_AUX1': [],
#     'RIGHT_AUX2': [],
#     'NO_AUX0': [],
#     'NO_AUX1': [],
#     'NO_AUX2': [],
#     'COUTER_AUX0': [],
#     'COUTER_AUX1': [],
#     'COUTER_AUX2': [],

def train_and_plot(times = 3, start = 0):
    epochs = 3
    ld_train = read_ld_train()
    ld_dev = read_ld_dev() 
    for model_idx_org in range(times):
        model_idx = model_idx_org + start
        SEED = RANDOM_SEEDs[model_idx]
        m = create_model_with_seed(SEED)
        cb = create_iteration_callback_shell(f'LEFT_AUX{model_idx}', m, ld_dev, test_left_aux, intensively_log_interval = 20)
        for i in range(epochs):
            train_left_aux(m, ld_train, fl_rate = 5.0, aux_rate = 0.1, iteration_callback = cb)
        m = create_model_with_seed(SEED)
        cb = create_iteration_callback_shell(f'RIGHT_AUX{model_idx}', m, ld_dev, test_right_aux, intensively_log_interval = 20)
        for i in range(epochs):
            train_right_aux(m, ld_train, fl_rate = 0, aux_rate = 0.2, iteration_callback = cb)
        m = create_model_with_seed(SEED)
        cb = create_iteration_callback_shell(f'NO_AUX{model_idx}', m, ld_dev, test_no_aux, intensively_log_interval = 20)
        for i in range(epochs):
            train_no_aux(m, ld_train, fl_rate = 2.0, iteration_callback = cb)
        m = create_model_with_seed(SEED)
        cb = create_iteration_callback_shell(f'COUTER_AUX{model_idx}', m, ld_dev, test_counter_aux, intensively_log_interval = 20)
        for i in range(epochs):
            train_counter_aux(m, ld_train, fl_rate = 0, iteration_callback = cb)
