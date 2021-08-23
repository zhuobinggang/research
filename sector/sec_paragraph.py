from sector_split import *
import mainichi_paragraph as custom_data


def targets(ld):
    targets = []
    for batches in ld:
        for ss, labels, pos in batches:
            if pos != 0:
                targets.append(labels[pos])
    return targets


# ==================== Auxiliary Methods =====================

def init_G_Symmetry_Mainichi(half=2, batch=1, mini=False):
    G['ld'] = custom_data.load_customized_loader(file_name='train',
                                                 half=half,
                                                 batch=batch,
                                                 mini=mini)
    G['testld'] = custom_data.load_customized_loader(file_name='test',
                                                     half=half,
                                                     batch=batch,
                                                     mini=mini)
    # G['validld'] = custom_data.load_customized_loader(file_name='valid',
    #                                                   half=half,
    #                                                   batch=batch,
    #                                                   mini=mini)
    G['devld'] = custom_data.load_customized_loader(file_name='dev',
                                                    half=half,
                                                    batch=batch,
                                                    mini=mini)


def fit_sigmoided_to_label_list(out):
    results = []
    for item in out:
        assert item >= 0 and item <= 1
        if item < 0.5:
            results.append(0) 
        else:
            results.append(1) 
    return results


def get_datas_early_stop_and_parameter_ajust_v2(epochs, dev_ld, desc = ''):
    dev_losses = []  # For early stop
    train_losses = []
    dics = []
    for i in range(epochs):
        train_losses += train_simple(G['m'], G['ld'], 1)
        dev_loss = cal_total_loss(G['m'], dev_ld)
        dev_losses.append(dev_loss)
        dic_i = get_test_result_dic(G['m'], dev_ld)
        dic_i['dev_loss'] = dev_loss  # Save index info
        dics.append(dic_i)
    dics[0]['desc'] = desc
    print(dics)
    G['mess_list'].append(dics)  # 将valid loss最小对应的dic放进mess_list

def get_datas_early_stop_and_parameter_ajust(index,
                                             epochs,
                                             desc,
                                             dic_to_send=None,
                                             url=None):
    dev_losses = []  # For early stop
    train_losses = []
    tested = []
    for i in range(epochs):
        train_losses += train_simple(G['m'], G['ld'], 1)
        dev_loss = cal_total_loss(G['m'], G['devld'])
        dev_losses.append(dev_loss)
        dic_to_analyse = get_test_result_dic(G['m'], G['testld'])
        dic_to_analyse['index'] = i  # Save index info
        dic_to_analyse['dev_loss'] = dev_loss  # Save index info
        dev_result_dic = get_test_result_dic(G['m'], G['devld'])
        dic_to_analyse['dev_result_dic'] = dev_result_dic
        tested.append(dic_to_analyse)
    test_result = tested[np.argmin(dev_losses)]
    print(test_result)
    # test_result['dev_losses'] = dev_losses
    G['mess_list'].append(test_result)  # 将valid loss最小对应的dic放进mess_list
    dic = test_result
    if dic_to_send is not None:
        dic = {**dic, **dic_to_send}
    else:
        pass
    R.request_my_logger(dic, desc, url)


def raw_atts_idss(m, ld, no_flat = False):
    atts = []
    idss = []
    results = []
    targets = []
    for mess in ld:
        atts_batch, idss_batch, result_batch, target_batch = m.get_att(mess)
        atts += atts_batch
        idss += idss_batch
        results += result_batch
        targets += target_batch
    if no_flat:
        return atts, idss, results, targets
    else:
        atts = B.flatten_num_lists(atts)
        idss = B.flatten_num_lists(idss)
        return atts, idss, results, targets

def project_to_dict(atts, idss):
    dic = {}
    for att, idx in zip(atts, idss):
        if dic.get(idx) is None:
            dic[idx] = att
        else: 
            dic[idx] += att
    return dic

def get_sorted_idx_att_pairs(dic):
    idx_att_pairs = dic.items()
    sorted_idx_att_pairs = list(reversed(sorted(idx_att_pairs, key = lambda x: x[1])))
    return sorted_idx_att_pairs

# Get most attended tokens
def get_most_attended_tokens(m):
    ld = G['testld']
    atts, idss, _, _ = raw_atts_idss(m, ld)
    dic = project_to_dict(atts, idss)
    sorted_idx_att_pairs = get_sorted_idx_att_pairs(dic)
    return [m.toker.decode(idx) for idx, _ in sorted_idx_att_pairs[:50]]

def filter_empty_list(lol):
    return [l for l in lol if len(l) > 0]

def result_dict_with_important_info(m):
    ld = G['testld']
    atts_batchs, idss_batchs, results, targets = raw_atts_idss(m, ld, no_flat = True)
    results = fit_sigmoided_to_label_list(results)
    atts_batchs = filter_empty_list(atts_batchs)
    idss_batchs = filter_empty_list(idss_batchs)
    assert len(idss_batchs) == len(results)
    # Filter true predicted guys
    atts = []
    idss = []
    for atts_batch, idss_batch, res, tar in zip(atts_batchs, idss_batchs, results, targets):
        if res == tar and tar == 1:
            atts.append(atts_batch)
            idss.append(idss_batch)
    atts = B.flatten_num_lists(atts_batchs)
    idss = B.flatten_num_lists(idss_batchs)
    dic = {}
    for att, idx in zip(atts, idss):
        if dic.get(idx) is None:
            dic[idx] = {'att': att, 'count': 1}
        else: 
            dic[idx]['att'] += att
            dic[idx]['count'] += 1
    # cal att/count
    for key in dic.keys():
        dic[key]['weighted_att'] = dic[key]['att'] / dic[key]['count']
    return dic

def get_useful_most_attended_tokens(m):
    dic = result_dict_with_important_info(m)
    # sort by weighted att
    twoples_id_weighted_att = []
    for key in dic.keys():
        twoples_id_weighted_att.append((key, dic[key]['weighted_att'], dic[key]['count']))
    twoples_id_weighted_att = list(reversed(sorted(twoples_id_weighted_att, key = lambda x: x[1])))
    return [m.toker.decode(idx) for idx, weighted_att, count in twoples_id_weighted_att[:100]]

def cal_exceed_rate(toker, arts):
    # toker = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    sentence_all = 4
    len_limit = int(500 / sentence_all)
    count = 0
    # arts = custom_data.read_sentences_per_art(f'datasets/train.paragraph.txt')
    for art in arts:
        for line in art:
            ids = toker.encode(line, add_special_tokens = False)
            if len(ids) > len_limit:
                print(line)
                count += 1
    return count


def get_att_ours(m, mass):
    sss, labels, poss = m.handle_mass(mass)
    atts = []
    idss = []
    results = []
    targets = []
    for ss, ls, pos in zip(sss, labels, poss):  # Different Batch
        if pos == 0:
            atts.append([])
            idss.append([])
            results.append(-1)
            targets.append(-1)
        else:
            (cls, att_cls), (seps,
                             att_seps), _, ids = B.compress_by_ss_then_pad(
                                 m.bert,
                                 m.toker,
                                 ss,
                                 pos,
                                 m.ss_len_limit,
                                 with_att=True)
            att = att_seps[pos]  # (token_count)
            atts.append(att.view(-1).tolist())
            idss.append(ids.view(-1).tolist())
            results.append(m.classifier(seps[pos - 1]).item())
            targets.append(ls[pos])
    return atts, idss, results, targets

# ONE CLS ONE SEP
def get_att_baseline(m, mass):
    sss, labels, poss = m.handle_mass(mass)
    atts = []
    idss = []
    results = []
    targets = []
    for ss, ls, pos in zip(sss, labels, poss):  # Different Batch
        if pos == 0:
            atts.append([])
            idss.append([])
            results.append(-1)
            targets.append(-1)
        else:
            cls, atts_from_cls, ids = B.compress_one_cls_one_sep_pool_cls_output_att(
                                 m.bert,
                                 m.toker,
                                 ss,
                                 pos)
            atts.append(atts_from_cls.view(-1).tolist())
            idss.append(ids.view(-1).tolist())
            results.append(m.classifier(cls.view(1, m.bert_size)).item())
            targets.append(ls[pos])
    return atts, idss, results, targets


def copy_to_chrome_console_then_render(toker, idss, atts):
    print([round(item,3) for item in atts])
    print([toker.decode(item) for item in idss])



# ================================== Auxiliary Methods ====================================


class Sec_Para(Sector_Split):
    def get_loss(self, mass):
        batch = len(mass)
        sss, labels, poss = self.handle_mass(mass)
        losses = []  # 所有batch内的part loss都flatten到这里来
        # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
        for ss, ls, pos in zip(sss, labels, poss):  # Different Batch
            # pos: number start from 0
            if pos == 0:
                pass  # Skip start row, no train no test
            else:
                cls, seps, _ = B.compress_by_ss_then_pad(
                    self.bert, self.toker, ss, pos, self.ss_len_limit)
                assert len(seps) == len(ss)
                # NOTE: 去头去尾，必要操作，因为对应是错位的
                # 根据pos对seps进行修整，对应到ls的数量
                seps = seps[:-1]  # 最后一个SEP不需要
                ls = ls[1:]  # 第一个label不需要
                pos = pos - 1  # POS也要调整，因为现在掐头去尾了，pos要-1 (比如ss[s]s的时候，2要变为1)，需要注意的是，pos=0的时候，会变为-1，在处理auxiliary loss的时候，要考虑到这一点
                # Convert ls to Tensor
                ls = t.LongTensor(ls).cuda()  # (ss_len), (0 or 1)
                assert ls.shape[0] == seps.shape[0]
                assert ls.shape[0] == len(ss) - 1
                # 稍微做一点tricky的事情，将其他loss(除了中间那个) * 0.5
                o = self.classifier(seps)  #(ss_len, 1)
                loss_part = [
                    self.cal_loss(o_item.view(1, 1), l_item.view(1))
                    for (o_item, l_item) in zip(o, ls)
                ]
                # NOTE: Augment loss at 'pos'
                loss_part = [(los * self.auxiliary_loss_rate if
                              (index != pos) else los)
                             for index, los in enumerate(loss_part)]
                losses += loss_part
        return losses

    @t.no_grad()
    def dry_run(self, mass):
        # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
        sss, labels, poss = self.handle_mass(mass)
        pos_outs = []
        pos_labels = []
        # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
        for ss, ls, pos in zip(sss, labels, poss):  # Different Batch
            # pos: number start from 0
            if pos == 0:
                pass  # Don’t run the first sentence because it is 100% correct
            else:
                cls, seps, _ = B.compress_by_ss_then_pad(
                    self.bert, self.toker, ss, pos, self.ss_len_limit)
                assert len(seps) == len(ss)
                ls = t.LongTensor(ls).cuda()  # (ss_len), (0 or 1)
                pos_outs.append(self.classifier(seps[pos - 1]).view(1))
                pos_labels.append(ls[pos])
        if len(pos_outs) < 1:
            return t.LongTensor([]), t.LongTensor([])
        else:
            pos_outs = t.stack(pos_outs).cuda()
            pos_labels = t.LongTensor(pos_labels).cuda()
            assert len(pos_outs.shape) == 2
            assert len(pos_labels.shape) == 1
            assert pos_outs.shape[0] == pos_labels.shape[0]
            return fit_sigmoided_to_label(pos_outs), pos_labels

    @t.no_grad()
    def get_att(self, mass):
        sss, labels, poss = self.handle_mass(mass)
        atts = []
        idss = []
        results = []
        targets = []
        for ss, ls, pos in zip(sss, labels, poss):  # Different Batch
            if pos == 0:
                atts.append([])
                idss.append([])
            else:
                (cls, att_cls), (seps,
                                 att_seps), _, ids = B.compress_by_ss_then_pad(
                                     self.bert,
                                     self.toker,
                                     ss,
                                     pos,
                                     self.ss_len_limit,
                                     with_att=True)
                att = att_seps[pos]  # (token_count)
                atts.append(att.view(-1).tolist())
                idss.append(ids.view(-1).tolist())
                results.append(self.classifier(seps[pos - 1]).item())
                targets.append(ls[pos])
        return atts, idss, results, targets



class Sec_Para_Standard_One_Sep_Use_Cls(Sector_Split):
    def get_loss(self, mass):
        batch = len(mass)
        sss, labels, poss = self.handle_mass(mass)
        losses = []  # 所有batch内的part loss都flatten到这里来
        for ss, ls, pos in zip(sss, labels, poss):  # Different Batch
            # pos: number start from 0
            if pos == 0:
                pass  # Don’t run the first sentence because it is 100% correct
            else:
                cls = B.compress_one_cls_one_sep_pool_cls(
                    self.bert, self.toker, ss, pos)
                # Convert ls to Tensor
                label = ls[pos]  # 第一个label不需要
                label = t.LongTensor([label]).cuda()  # (ss_len), (0 or 1)
                o = self.classifier(cls.view(1, self.bert_size))  #(1, 1)
                losses.append(self.cal_loss(o.view(1, 1), label.view(1)))
        return losses

    @t.no_grad()
    def dry_run(self, mass):
        # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
        sss, labels, poss = self.handle_mass(mass)
        pos_outs = []
        pos_labels = []
        # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
        for ss, ls, pos in zip(sss, labels, poss):  # Different Batch
            if pos == 0:
                pass  # Don’t run the first sentence because it is 100% correct
            else:
                # pos: number start from 0
                cls = B.compress_one_cls_one_sep_pool_cls(
                    self.bert, self.toker, ss, pos)
                # Convert ls to Tensor
                o = self.classifier(cls.view(1, self.bert_size))  #(1, 1)
                pos_outs.append(o.view(1))
                pos_labels.append(ls[pos])
        if len(pos_outs) < 1:
            return t.LongTensor([]), t.LongTensor([])
        else:
            pos_outs = t.stack(pos_outs)
            pos_labels = t.LongTensor(pos_labels).cuda()
            assert len(pos_outs.shape) == 2
            assert len(pos_labels.shape) == 1
            assert pos_outs.shape[0] == pos_labels.shape[0]
            return fit_sigmoided_to_label(pos_outs), pos_labels


# ================================== Model ====================================


def sec_para_panther():
    panther_url = 'https://hookb.in/VGERm7dJyjtE22bwzZ7d'
    init_G_Symmetry_Mainichi(half=2, batch=4, mini=False)
    for i in range(20):
        G['m'] = m = Sec_Para(learning_rate=5e-6,
                              ss_len_limit=4,
                              auxiliary_loss_rate=0.2)
        get_datas_early_stop_and_parameter_ajust(
            i,
            3,
            f'Early Stop, Dev Ajust, Auxiliary Rate = {m.auxiliary_loss_rate}',
            url=panther_url)


def sec_para_pc():
    panther_url = 'https://hookb.in/VGERm7dJyjtE22bwzZ7d'
    init_G_Symmetry_Mainichi(half=2, batch=4, mini=False)
    for i in range(20):
        G['m'] = m = Sec_Para(learning_rate=5e-6,
                              ss_len_limit=4,
                              auxiliary_loss_rate=0.8)
        get_datas_early_stop_and_parameter_ajust(
            i,
            3,
            f'Early Stop, Dev Ajust, Auxiliary Rate = {m.auxiliary_loss_rate}',
            url=panther_url)
        G['m'] = m = Sec_Para(learning_rate=5e-6,
                              ss_len_limit=4,
                              auxiliary_loss_rate=1.0)
        get_datas_early_stop_and_parameter_ajust(
            i,
            3,
            f'Early Stop, Dev Ajust, Auxiliary Rate = {m.auxiliary_loss_rate}',
            url=panther_url)


def sec_para_zero_rate():
    panther_url = 'https://hookb.in/VGERm7dJyjtE22bwzZ7d'
    init_G_Symmetry_Mainichi(half=2, batch=4, mini=False)
    for i in range(20):
        G['m'] = m = Sec_Para(learning_rate=5e-6,
                              ss_len_limit=4,
                              auxiliary_loss_rate=0.0)
        get_datas_early_stop_and_parameter_ajust(
            i,
            3,
            f'Early Stop, Dev Ajust, Auxiliary Rate = {m.auxiliary_loss_rate}',
            url=panther_url)

def sec_para_standard_1vs1():
    panther_url = 'https://hookb.in/VGERm7dJyjtE22bwzZ7d'
    init_G_Symmetry_Mainichi(half=1, batch=4, mini=False)
    for i in range(20):
        G['m'] = m = Sec_Para_Standard_One_Sep_Use_Cls(
            learning_rate=5e-6, ss_len_limit=2, auxiliary_loss_rate=-1.0)
        get_datas_early_stop_and_parameter_ajust(i,
                                                 3,
                                                 f'Early Stop, Standard 1vs1',
                                                 url=panther_url)


def sec_para_standard_win6():
    panther_url = 'https://hookb.in/VGERm7dJyjtE22bwzZ7d'
    init_G_Symmetry_Mainichi(half=3, batch=4, mini=False)
    for i in range(20):
        G['m'] = m = Sec_Para_Standard_One_Sep_Use_Cls(
            learning_rate=5e-6, ss_len_limit=6, auxiliary_loss_rate=-1.0)
        get_datas_early_stop_and_parameter_ajust(i,
                                                 3,
                                                 f'Standard window size = 6',
                                                 url=panther_url)




def sec_para_standard(ld, max_train_epoch = 3):
    if G.get('ld') is None:
        init_G_Symmetry_Mainichi(half=2, batch=4, mini=False)
    for i in range(10):
        G['m'] = m = Sec_Para_Standard_One_Sep_Use_Cls(
            learning_rate=5e-6, ss_len_limit=4, auxiliary_loss_rate=-1.0)
        get_datas_early_stop_and_parameter_ajust_v2(max_train_epoch, ld, f'Early Stop, Standard')

def run_FL(ld, fl_rate = 0, max_train_epoch = 3):
    if G.get('ld') is None:
        init_G_Symmetry_Mainichi(half=2, batch=4, mini=False)
    for i in range(10):
        G['m'] = m = Sec_Para_Standard_One_Sep_Use_Cls(
            learning_rate=5e-6, ss_len_limit=4, auxiliary_loss_rate=-1.0, fl_rate = fl_rate)
        get_datas_early_stop_and_parameter_ajust_v2(max_train_epoch, ld, f'Early Stop, Dev Ajust, HP: fl_rate = {fl_rate}')

def sec_para_rate(ld, rate=0.0,max_train_epoch = 3):
    if G.get('ld') is None:
        init_G_Symmetry_Mainichi(half=2, batch=4, mini=False)
    for i in range(10):
        G['m'] = m = Sec_Para(learning_rate=5e-6,
                              ss_len_limit=4,
                              auxiliary_loss_rate=rate)
        print(f'Early Stop, Dev Ajust, Auxiliary Rate = {m.auxiliary_loss_rate}')
        get_datas_early_stop_and_parameter_ajust_v2(max_train_epoch, ld, f'Early Stop, Dev Ajust, Auxiliary Rate = {m.auxiliary_loss_rate}')


def rate_test_on_pc():
    for rate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        sec_para_rate(rate)


def rate_test_on_panther():
    sec_para_standard()
    for fl_rate in [1.0, 2.0, 5.0, 0.5, 0.0]:
        run_FL(fl_rate)
    for rate in [1.1, 1.2, 1.3, 1.4, 1.5]:
        sec_para_rate(rate)


def the_last_run():
    init_G_Symmetry_Mainichi(half=2, batch=4, mini=False)
    # 根据epoch数跑standard
    sec_para_standard(G['testld'], 2)
    # 根据auxiliary rate和epoch数跑实验
    sec_para_rate(G['testld'], 0.1, 3)
    # 根据auxiliary rate和epoch数跑实验
    run_FL(G['testld'], 1.0, 2)


def train_and_save(start_index = 0):
    init_G_Symmetry_Mainichi(half=2, batch=4, mini=False)
    # My, epoch = 3
    for i in range(5):
        G['m'] = m = Sec_Para(learning_rate=5e-6,
                              ss_len_limit=4,
                              auxiliary_loss_rate=0.1)
        train_simple(G['m'], G['ld'], 1)
        train_simple(G['m'], G['ld'], 1)
        train_simple(G['m'], G['ld'], 1)
        t.save(m, f'save/my_{i + start_index}.tch')
    # Standard, epoch = 2
    for i in range(5):
        G['m'] = m = Sec_Para_Standard_One_Sep_Use_Cls(
            learning_rate=5e-6, ss_len_limit=4, auxiliary_loss_rate=-1.0)
        train_simple(G['m'], G['ld'], 1)
        train_simple(G['m'], G['ld'], 1)
        t.save(m, f'save/stand_{i + start_index}.tch')
    # FL, epoch = 2
    for i in range(5):
        G['m'] = m = Sec_Para_Standard_One_Sep_Use_Cls(
            learning_rate=5e-6, ss_len_limit=4, auxiliary_loss_rate=-1.0, fl_rate = 1.0)
        train_simple(G['m'], G['ld'], 1)
        train_simple(G['m'], G['ld'], 1)
        t.save(m, f'save/fl_{i + start_index}.tch')


# EXP

def exp():
    init_G_Symmetry_Mainichi(half=2, batch=4, mini=False)
    m1 = t.load('save/my_0.tch')
    m2 = t.load('save/fl_0.tch')
    tld = G['testld']
    dic1 = {'att': [], 'idss': [], 'results': [], 'targets': []}
    dic2 = {'att': [], 'idss': [], 'results': [], 'targets': []}
    for mass in tld:
        atts, idss, results, targets = get_att_ours(m1, mass)
        dic1['att'] += atts
        dic1['idss'] += idss
        dic1['results'] += results
        dic1['targets'] += targets
        atts, idss, results, targets = get_att_baseline(m2, mass)
        dic2['att'] += atts
        dic2['idss'] += idss
        dic2['results'] += results
        dic2['targets'] += targets
    id_value_pairs = []
    for i in range(len(dic1['results'])):
        if np.abs(dic1['targets'][i] - dic1['results'][i]) < 0.3 and np.abs(dic2['targets'][i] - dic2['results'][i]) > 0.7:
            id_value_pairs.append((i, np.abs(dic1['results'][i] - dic2['results'][i])))
    # return id_value_pairs, dic1, dic2
    # SHOW
    idss = dic1['idss'][388]
    att1 = dic1['att'][388]
    att2 = dic2['att'][388]
    for idx, a1, a2 in zip(idss, att1, att2):
        print(f'{m1.toker.decode(idx)}: {round(a1, 3)}, {round(a2, 3)}')
            






