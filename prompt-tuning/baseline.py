from transformers import BertJapaneseTokenizer, BertForSequenceClassification
from trainable_prefix_abstract import *
from reader import read_ld_train_from_chapters, read_ld_test_from_chapters, read_ld_dev_from_chapters
import torch
import datetime
import numpy as np
import types

def create_model(learning_rate = 1e-5):
    res = types.SimpleNamespace()
    # NOTE
    res.bert = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    res.toker = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    res.opter = torch.optim.AdamW(res.bert.parameters(), learning_rate)
    res.bert.cuda()
    res.bert.train()
    return res

def cal_loss(m, item):
    ss, ls = item
    ids = m.toker.encode(f'[CLS]{ss[0]}[SEP]{ss[1]}[SEP]', add_special_tokens = False)
    ids = torch.LongTensor([ids])
    loss = m.bert(ids.cuda(), labels = torch.LongTensor([ls[1]]).cuda()).loss
    return loss

def dry_run(m, item):
    ss, ls = item
    true_y = ls[1]
    ids = m.toker.encode(f'[CLS]{ss[0]}[SEP]{ss[1]}[SEP]', add_special_tokens = False)
    ids = torch.LongTensor([ids])
    with torch.no_grad():
        logits = m.bert(ids.cuda()).logits
    pred = logits.argmax().item()
    return pred, true_y


def learning_curve(epochs = 12, batch = 16):
    m = create_model()
    m.bert.requires_grad_(True)
    train_ds_org = read_ld_train_from_chapters(2)
    np.random.shuffle(train_ds_org) # NOTE: 注意是否shuffle
    train_ds_4_splits = [train_ds_org[:7000], train_ds_org[7000:14000], train_ds_org[14000: 21000], train_ds_org[21000: 28000]]
    test_ds = read_ld_test_from_chapters(2)
    opter = m.opter
    path = 'baseline_curve.png'
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
    x = list(range(epochs * 4))
    scale_down_batch_losses = [loss / 10 for loss in batch_losses]
    draw_line_chart(x, [scale_down_batch_losses, precs, recs, fs, fake_fs], ['batch loss', 'precs', 'recs', 'fs', 'random fs'], path = path, colors = ['r','g','b','y', 'k'])
    return m, [scale_down_batch_losses, precs, recs, fs, fake_fs]

def early_stop_n_times(restart = 5, epoch = 1):
    train_ds = read_ld_train_from_chapters(2)
    test_ds = read_ld_test_from_chapters(2)
    ress = []
    for _ in range(restart):
        m = create_model()
        m.bert.requires_grad_(True)
        opter = m.opter
        for e in range(epoch):
            _ = train_one_epoch(m, train_ds, cal_loss, opter, batch = 16, log_inteval = 1000)
        ress.append(test(m, test_ds, dry_run))
    return ress
