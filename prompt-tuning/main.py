from reader import read_ld_train_from_chapters, read_ld_test_from_chapters, read_ld_dev_from_chapters
from trainable_prefix_abstract import train_one_epoch, get_emb_without_position_info_for_concat, embedding_with_position_info, bert_encoder, create_model_with_trainable_prefix, loss_by_emb_without_position_info, predict_by_emb_without_position_info, test, draw_line_chart
import numpy as np
import torch

def get_inputs_emb_without_pos_info(m, ss):
    cls_emb = get_emb_without_position_info_for_concat(m, '[CLS]') # (1, 1, 768)
    prefix_emb = m.trainable_prefixs # (1, 10, 768)
    mask_emb = get_emb_without_position_info_for_concat(m, '[MASK]') # (1, 1, 768)
    s0_emb = get_emb_without_position_info_for_concat(m, ss[0]) # (1, ?, 768)
    sep0_emb = get_emb_without_position_info_for_concat(m, '[SEP]') # (1, 1, 768)
    s1_emb = get_emb_without_position_info_for_concat(m, ss[1]) # (1, ?, 768)
    sep1_emb = get_emb_without_position_info_for_concat(m, '[SEP]') # (1, 1, 768)
    inputs_emb_without_pos_info = torch.cat([cls_emb, prefix_emb, mask_emb, s0_emb, sep0_emb, s1_emb, sep1_emb], dim = 1)
    # after concat
    mask_index = cls_emb.shape[1] + prefix_emb.shape[1]
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

# m = create_model_with_trainable_prefix('分割', 10, 1e-3)
# ld_train = read_ld_train_from_chapters(2)
# ld_test = read_ld_train_from_chapters(2)
# batch_losses = train_one_epoch(m, ld_train, cal_loss, m.opter_prefixs, batch = 4)

def learning_curve(prompt_tuning_only = True, epochs = 5, batch = 8, lr = 2e-5, prefix_length = 10):
    if prompt_tuning_only:
        m = create_model_with_trainable_prefix('分割', prefix_length, lr)
        m.bert.requires_grad_(False)
        m.trainable_prefixs.requires_grad_(True)
        opter = m.opter_prefixs
        path = 'learning_curve_prompt_tuning_only.png'
    else:
        print('!!!!!! Not finished !!!!!!')
        return None
        m = create_model_with_trained_prefix('trained_prefix_len10_epoch10.tch')
        # m.trainable_prefixs.requires_grad_(False)
        opter = m.opter
        path = 'learning_curve_prompt_tuning_finetune.png'
    train_ds = read_ld_train_from_chapters(2)
    test_ds = read_ld_train_from_chapters(2)
    batch_losses = []
    precs = []
    recs = []
    fs = []
    fake_fs = []
    for e in range(epochs):
        batch_losses.append(np.mean(train_one_epoch(m, train_ds, cal_loss, opter, batch = batch, log_inteval = 1000)))
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


def early_stop_5_times(prompt_tuning_only = True, restart = 5, epoch = 2):
    train_ds = read_ld_train_from_chapters(2)
    test_ds = read_ld_train_from_chapters(2)
    ress = []
    for _ in range(restart):
        if prompt_tuning_only:
            m = create_model_with_trainable_prefix('分割', 10, 1e-3)
            opter = m.opter_prefixs
            m.bert.requires_grad_(False)
            m.trainable_prefixs.requires_grad_(True)
        else:
            m = create_model_with_trained_prefix('trained_prefix_len10_epoch10.tch')
            m.trainable_prefixs.requires_grad_(False)
            opter = m.opter
        for e in range(epoch):
            _ = train_one_epoch(m, train_ds, cal_loss, opter, batch = 8)
        ress.append(test(m, test_ds, dry_run))
    return ress


def train_and_store(restart = 10):
    train_ds = read_ld_train_from_chapters(2)
    test_ds = read_ld_train_from_chapters(2)
    ress = []
    m = create_model_with_trainable_prefix('分割', 10, 1e-4)
    m.bert.requires_grad_(False)
    m.trainable_prefixs.requires_grad_(True)
    for _ in range(restart):
        _ = train_one_epoch(m, train_ds, cal_loss, m.opter_prefixs, batch = 8)
        ress.append(test(m, test_ds, dry_run))
    with torch.no_grad():
        torch.save(m.trainable_prefixs, 'trained_prefix_len10_epoch10.tch')



