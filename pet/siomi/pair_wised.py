from trainable_prefix_abstract import *
from reader import read_five_splits
import random
import numpy as np


def run():
    m = create_model()
    dss = read_five_splits()
    for train_ds, test_ds, dev_ds in dss:
        _ = train_one_epoch(m, train_ds, cal_loss, m.opter, batch = 4, log_inteval = 4)
        prec, rec, f, fake_f = test(m, test_ds, dry_run, need_random_baseline = True)


def get_inputs_emb_without_pos_info(m, left, right):
    emb_before_mask = get_emb_without_position_info_for_concat(m, f'[CLS]「{left}」は「{right}」よりも上品ですか？[SEP]')
    emb_after_mask = get_emb_without_position_info_for_concat(m, f'[MASK]。[SEP]')
    inputs_emb_without_pos_info = torch.cat([emb_before_mask, emb_after_mask], dim = 1)
    mask_index = emb_before_mask.shape[1]
    return inputs_emb_without_pos_info, mask_index

def cal_loss(m, item):
    good, bad = item
    true_y = random.randint(0, 1)
    if true_y == 1:
        inputs_emb_without_pos_info, mask_index = get_inputs_emb_without_pos_info(m, good, bad)
    else:
        inputs_emb_without_pos_info, mask_index = get_inputs_emb_without_pos_info(m, bad, good)
    loss = loss_by_emb_without_position_info(m, inputs_emb_without_pos_info, mask_index, label = true_y)
    return loss

def dry_run(m, item, random_left_right = True):
    good, bad = item
    if random_left_right:
        true_y = random.randint(0, 1)
        if true_y == 1:
            inputs_emb_without_pos_info, mask_index = get_inputs_emb_without_pos_info(m, good, bad)
        else:
            inputs_emb_without_pos_info, mask_index = get_inputs_emb_without_pos_info(m, bad, good)
    else: 
        true_y = 1
        inputs_emb_without_pos_info, mask_index = get_inputs_emb_without_pos_info(m, good, bad)
    word = predict_by_emb_without_position_info(m, inputs_emb_without_pos_info, mask_index)
    word = word.replace(' ', '')
    if word == 'はい':
        return 1, true_y
    elif word == 'いえ':
        return 0, true_y
    else:
        print(f'Turned word {word} to zero!')
        return 0, true_y

def get_predicted_word(m, good, bad):
    inputs_emb_without_pos_info, mask_index = get_inputs_emb_without_pos_info(m, good, bad)
    word = predict_by_emb_without_position_info(m, inputs_emb_without_pos_info, mask_index)
    return word



#     m = create_model()
#     dss = read_five_splits()
#     for train_ds, test_ds, dev_ds in dss:
#         _ = train_one_epoch(m, train_ds, cal_loss, m.opter, batch = 4, log_inteval = 4)
#         prec, rec, f, fake_f = test(m, test_ds, dry_run, need_random_baseline = True)

def learning_curve(epochs = 12, batch = 4):
    m = create_model()
    m.bert.requires_grad_(True)
    dss = read_five_splits()
    train_ds, test_ds, dev_ds = dss[0]
    opter = m.opter
    path = 'learning_curve_manual.png'
    batch_losses = []
    precs = []
    recs = []
    fs = []
    fake_fs = []
    for e in range(epochs):
        batch_losses.append(np.mean(train_one_epoch(m, train_ds, cal_loss, opter, batch = batch)))
        prec, rec, f, fake_f = test(m, test_ds, dry_run, need_random_baseline = True)
        precs.append(prec)
        recs.append(rec)
        fs.append(f)
        fake_fs.append(fake_f)
    x = list(range(epochs))
    # scale_down_batch_losses = [loss / 10 for loss in batch_losses]
    scale_down_batch_losses = batch_losses
    draw_line_chart(x, [scale_down_batch_losses, precs, recs, fs, fake_fs], ['batch loss', 'precs', 'recs', 'fs', 'random fs'], path = path, colors = ['r','g','b','y', 'k'])
    return m, [scale_down_batch_losses, precs, recs, fs, fake_fs]

def grid_search(epochs = 10, batch = 8):
    dss = read_five_splits()
    res = []
    for train_ds, test_ds, dev_ds in dss:
        per_ds = []
        for _ in range(5):
            m = create_model()
            m.bert.requires_grad_(True)
            opter = m.opter
            per_model = []
            for e in range(epochs):
                _ = train_one_epoch(m, train_ds, cal_loss, opter, batch = batch)
                prec, rec, f = test(m, dev_ds, dry_run, need_random_baseline = False)
                per_model.append(f)
            per_ds.append(per_model)
        res.append(per_ds)
    return np.array(res)

def show_grid_search_result():
    dd = np.load('grid_search.npy')
    print(dd.shape) # 5,5,10 = (dataset, model, epoch)
    print(dd.mean(1).argmax(1))

def dry_run_no_shuffle(m, item):
    return dry_run(m, item, random_left_right = False)

def test_by_parameters(epochs = [4, 6, 5, 3, 1], batch = 8, shuffle = False):
    np.random.seed(1)
    dss = read_five_splits()
    res = []
    for dataset_index, (train_ds, test_ds, _) in enumerate(dss):
        per_ds = []
        for model_index in range(5):
            m = create_model()
            m.bert.requires_grad_(True)
            opter = m.opter
            for e in range(epochs[dataset_index] + 1):
                _ = train_one_epoch(m, train_ds, cal_loss, opter, batch = batch)
            print(f'dataset index: {dataset_index}, model index: {model_index}, trained epochs: {epochs[dataset_index] + 1}')
            if shuffle:
                prec, rec, f = test(m, test_ds, dry_run, need_random_baseline = False)
            else:
                prec, rec, f = test(m, test_ds, dry_run_no_shuffle, need_random_baseline = False)
            per_ds.append((prec, rec, f))
        res.append(per_ds)
    return np.array(res)

