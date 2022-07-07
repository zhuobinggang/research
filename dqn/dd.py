
def pad_idss(idss):
    PAD_TOKEN_ID = 0
    max_length = np.max([len(ids) for ids in idss])
    idss_padded = []
    masks = []
    if hasattr(idss[0], 'tolist'):
        idss = [ids.tolist() for ids in idss]
    for ids in idss:
        idss_padded.append(ids + [0] * (max_length - len(ids)))
        masks.append([1] * len(ids) + [0] * (max_length - len(ids)))
    return idss_padded, masks


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def batched_random_dataset(dataset, batch_size = 4):
    return batch([(row_idx, row) for row_idx, row in enumerate(np.random.permutation(dataset))], batch_size)


def dd(train, m):
    idss = []
    headword_indexss = []
    for i in range(4):
        row = train[i]
        tokens_org = row['tokens']
        tokens, ids, headword_indexs = subword_tokenize(tokens_org, m.toker)
        idss.append(ids.squeeze())
        headword_indexss.append(headword_indexs)
        # out_bert = bert(ids.cuda()).last_hidden_state[:, headword_indexs, :] # (1, n, 768)

