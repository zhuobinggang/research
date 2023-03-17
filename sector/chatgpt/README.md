## Zero shot news

res: (0.28, 0.23333333333333334, 0.2545454545454545, 0.48809523809523814)

```py
from manual_exp_news import *
from chatGPT import send, create_request_content
contents = create_request_content(transfer_mld(mld))
ress = []
for content in contents:
    ress.append(send(content, role='system', max_tokens=100))
# Transfer responses to labels
responses = [item['choices'][0]['message']['content'] for item in ress]
zero_ones = []
for text in responses:
    temp = text.strip().lower()
    if temp.startswith('no'):
        zero_ones.append(0)
    elif temp.startswith('yes'):
        zero_ones.append(1)
    else:
        print(f'Could not prossess by program: {temp}')
        zero_ones.append(temp)
from utils_lite import cal_prec_rec_f1_v2
labels = [ls[2] for ss, ls in transfer_mld(mld)]
cal_prec_rec_f1_v2(zero_ones, labels)

import pickle
txt = open('manual_exp/news_manual_dataset_chatgpt_responses.txt','w')
pickle.dump(ress,txt)
txt.close()

```

## Zero shot novel

```py
from manual_exp_novel import *
from chatGPT import send, create_request_content
dstest = read_ld_test()
idxs = random_100_index()
mld = select_from_dstest(dstest, idxs)
contents = create_request_content(mld)
ress = []
for content in contents:
    ress.append(send(content, role='system', max_tokens=100))
responses = [item['choices'][0]['message']['content'] for item in ress]
zero_ones = []
for text in responses:
    temp = text.strip().lower()
    if temp.startswith('no'):
        zero_ones.append(0)
    elif temp.startswith('yes'):
        zero_ones.append(1)
    else:
        print(f'Could not prossess by program: {temp}')
        zero_ones.append(temp)
labels = [ls[2] for ss, ls in mld]
from utils_lite import cal_prec_rec_f1_v2
cal_prec_rec_f1_v2(zero_ones, labels)

import pickle
txt = open('manual_exp/novel_manual_dataset_chatgpt_responses.txt','wb')
pickle.dump(ress,txt)
txt.close()
```
### CoT

```py
from manual_exp_novel import *
from chatGPT import send, create_request_content_cot
dstest = read_ld_test()
idxs = random_100_index()
mld = select_from_dstest(dstest, idxs)
contents = create_request_content_cot(mld)
ress = []
for content in contents:
    ress.append(send(content))
responses = [item['choices'][0]['message']['content'] for item in ress]
zero_ones = []
for text in responses:
    temp = text.strip().lower()
    if temp.startswith('no'):
        zero_ones.append(0)
    elif temp.startswith('yes'):
        zero_ones.append(1)
    else:
        print(f'Could not prossess by program: {temp}')
        zero_ones.append(temp)
labels = [ls[2] for ss, ls in mld]
from utils_lite import cal_prec_rec_f1_v2
cal_prec_rec_f1_v2(zero_ones, labels)

import pickle
txt = open('manual_exp/novel_manual_dataset_chatgpt_responses.txt','wb')
pickle.dump(ress,txt)
txt.close()
```



