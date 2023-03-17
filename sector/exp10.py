# 修论中的例子输出结果
# 小说: My PC
from exp_novel import load_model, read_ld_test
dstest = read_ld_test()
dt = dstest[774:807]
# AUXFL
m = load_model(f'SEED_{97}_AUX01FL50E2_{5}')
y_true, y_pred = test(dt, m)
y_pred_rounded = [(1 if y > 0.5 else 0) for y in y_pred]
text = ''
for item, y_pred in zip(dt, y_pred_rounded):
    ss, ls = item
    label = ls[2]
    if label == 1:
        text += '<br/> <br/>'
    text += f'【{y_pred}】{ss[2]} '
# FL
m = load_model(f'SEED_{97}_FL50E2_{5}')
y_true, y_pred = test_baseline(dt, m)
y_pred_rounded = [(1 if y > 0.5 else 0) for y in y_pred]
text = ''
for item, y_pred in zip(dt, y_pred_rounded):
    ss, ls = item
    label = ls[2]
    if label == 1:
        text += '<br/> <br/> '
    text += f'【{y_pred}】{ss[2]} '


# 新闻: Panther
from exp_news import *
dstest = read_ld_tests()[0]
dt = dstest[176:189]
# AUXFL
m = load_model(f'SEED{97}_AUX01FL50E2')
y_true, y_pred = test(dt, m)
y_pred_rounded = [(1 if y > 0.5 else 0) for y in y_pred]
text = ''
for item, y_pred in zip(dt, y_pred_rounded):
    ss, ls = item
    label = ls[2]
    if label == 1:
        text += '<br/> <br/>'
    text += f'【{y_pred}】{ss[2]} '
# FL
m = load_model(f'SEED{97}_FL20E3')
y_true, y_pred = test_baseline(dt, m)
y_pred_rounded = [(1 if y > 0.5 else 0) for y in y_pred]
text = ''
for item, y_pred in zip(dt, y_pred_rounded):
    ss, ls = item
    label = ls[2]
    if label == 1:
        text += '<br/> <br/>'
    text += f'【{y_pred}】{ss[2]} '

    
