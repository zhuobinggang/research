from rainable_prefix_abstract import *
from reader import read_five_splits


def cal_loss(m, item):
    good, bad = item
    pass

m = create_model()
dss = read_five_splits()
for train_ds, test_ds in dss:
    train_one_epoch(m, train_ds, cal_loss, opter, batch = 4, log_inteval = 4)


def get_inputs_emb_without_pos_info(m, left, right):
    emb_before_mask = get_emb_without_position_info_for_concat(m, f'[CLS]「{left}」は「{right}」よりも上品ですか？[SEP]')
    emb_after_mask = get_emb_without_position_info_for_concat(m, f'[MASK]。[SEP]')
    inputs_emb_without_pos_info = torch.cat([emb_before_mask, emb_after_mask], dim = 1)
    mask_index = emb_before_mask.shape[1]
    return inputs_emb_without_pos_info, mask_index

def cal_loss(m, item):
    good, bad = item
    inputs_emb_without_pos_info, mask_index = get_inputs_emb_without_pos_info(m, good, bad)
    loss1 = loss_by_emb_without_position_info(m, inputs_emb_without_pos_info, mask_index, label = 1)
    inputs_emb_without_pos_info, mask_index = get_inputs_emb_without_pos_info(m, bad, good)
    loss2 = loss_by_emb_without_position_info(m, inputs_emb_without_pos_info, mask_index, label = 0)
    return loss1 + loss2

def dry_run(m, item):
    good, bad = item
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


def learning_curve(epochs = 12, batch = 4):
    m = create_model()
    m.bert.requires_grad_(True)
    train_ds, test_ds = customized_ds()
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
    scale_down_batch_losses = [loss / 10 for loss in batch_losses]
    draw_line_chart(x, [scale_down_batch_losses, precs, recs, fs, fake_fs], ['batch loss', 'precs', 'recs', 'fs', 'random fs'], path = path, colors = ['r','g','b','y', 'k'])
    return m, [scale_down_batch_losses, precs, recs, fs, fake_fs]
