from trainable_prefix_abstract import *
from reader import read_ld_train_from_chapters, read_ld_test_from_chapters, read_ld_dev_from_chapters
import numpy as np
import torch

def get_inputs_emb_without_pos_info(m, ss):
    before_mask = get_emb_without_position_info_for_concat(m, f'[CLS]{ss[0]}[SEP]（新しい段落ですか？') # (1, 1, 768)
    after_mask = get_emb_without_position_info_for_concat(m, f'[MASK]）{ss[1]}[SEP]')
    inputs_emb_without_pos_info = torch.cat([before_mask, after_mask], dim = 1)
    # after concat
    mask_index = before_mask.shape[1]
    return inputs_emb_without_pos_info, mask_index

def cal_loss(m, item):
    ss, ls = item
    inputs_emb_without_pos_info, mask_index = get_inputs_emb_without_pos_info(m, ss)
    return loss_by_emb_without_position_info(m, inputs_emb_without_pos_info, mask_index, ls[1])

def dry_run(m, item):
    ss, ls = item
    true_y = ls[1]
    inputs_emb_without_pos_info, mask_index = get_inputs_emb_without_pos_info(m, ss)
    word = predict_by_emb_without_position_info(m, inputs_emb_without_pos_info, mask_index)
    word = word.replace(' ', '')
    if word == 'はい':
        return 1, true_y
    elif word == 'いえ':
        return 0, true_y
    else:
        print(f'Turned word {word} to zero!')
        return 0, true_y

def get_predicted_word(m, item):
    ss, _ = item
    inputs_emb_without_pos_info, mask_index = get_inputs_emb_without_pos_info(m, ss)
    word = predict_by_emb_without_position_info(m, inputs_emb_without_pos_info, mask_index)
    return word

def learning_curve(epochs = 12, batch = 16):
    m = create_model()
    m.bert.requires_grad_(True)
    train_ds_org = read_ld_train_from_chapters(2)
    train_ds_4_splits = [train_ds_org[:7000], train_ds_org[7000:14000], train_ds_org[14000: 21000], train_ds_org[21000: 28000]]
    test_ds = read_ld_test_from_chapters(2)
    opter = m.opter
    path = 'manual_prompt.png'
    batch_losses = []
    precs = []
    recs = []
    fs = []
    fake_fs = []
    for e in range(epochs):
        for ds in train_ds_4_splits:
            batch_losses.append(np.mean(train_one_epoch(m, ds, cal_loss, opter, batch = batch, log_inteval = 1000)))
            prec, rec, f, fake_f = test(m, test_ds, dry_run, need_random_baseline = True)
            precs.append(prec)
            recs.append(rec)
            fs.append(f)
            fake_fs.append(fake_f)
    x = list(range(epochs))
    scale_down_batch_losses = [loss / 10 for loss in batch_losses]
    draw_line_chart(x, [scale_down_batch_losses, precs, recs, fs, fake_fs], ['batch loss', 'precs', 'recs', 'fs', 'random fs'], path = path, colors = ['r','g','b','y', 'k'])
    return m, [scale_down_batch_losses, precs, recs, fs, fake_fs]

def early_stop_n_times(restart = 10, epoch = 1):
    train_ds, test_ds = customized_ds()
    ress = []
    for _ in range(restart):
        m = create_model()
        m.bert.requires_grad_(True)
        opter = m.opter
        for e in range(epoch):
            _ = train_one_epoch(m, train_ds, cal_loss, opter, batch = 16, log_inteval = 1000)
        ress.append(test(m, test_ds, dry_run))
    return ress

def dd():
    _, res_curve = learning_curve(epochs = 5, batch = 16)
    res_list = early_stop_n_times(10, 1)
    return res_curve, res_list

