

def run():
  for row_idx, row in enumerate(np.random.permutation(ds_train)):
      if row_idx % 1000 == 0:
          # print(f'finished: {row_idx}/{len(ds_train)}')
          pass
      tokens_org = row['tokens']
      # DESC: 每个token可能会被分解成多个subword，所以用headword_indexs来获取开头的subword对应的embedding
      tokens, ids, headword_indexs = subword_tokenize(tokens_org, m.toker)
      if tokens is None:
          print('跳过训练')
      else:
          out_bert = bert(ids.cuda()).last_hidden_state[:, headword_indexs, :] # (1, n, 768)
          
          out_lstm, _ = m.lstm(out_bert) # (1, n, 256 * 2)
          out_mlp = m.mlp(out_lstm) # (1, n, 9)
          ys = F.softmax(out_mlp, dim = 2) # (1, n, 9)
          # cal loss
          labels = t.LongTensor(row['ner_tags']) # Long: (n)
          loss = CEL(ys.squeeze(0), labels.cuda())
          # backward
          m.zero_grad()
          loss.backward()
          opter.step()


def logic():
    for i in range(len(token_embs))
        done = (i == len(token_embs) - 1)
        now_state = token_embs[i]
        if should_explore:
            action = sample()
        else:
            action = policy(now_state)
        reward = get_reward(action, i)
        q_pred = Q(now_state, action)
        q_true = Q_max(token_embs[i + 1]) + reward
        loss = BCE(q_pred, q_true)



