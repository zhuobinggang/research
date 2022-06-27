import torch as t
import torch.optim as optim
from torchcrf import CRF

trans_BOS = t.distributions.Categorical(t.tensor([8., 2]))
trans0 = t.distributions.Categorical(t.tensor([8., 2]))
trans1 = t.distributions.Categorical(t.tensor([6., 4]))
trans = [trans0, trans1]
emiss0 = t.distributions.Categorical(t.tensor([6,2,2.]))
emiss1 = t.distributions.Categorical(t.tensor([2,4,4.]))
emiss = [emiss0, emiss1]

def sample(steps = 5): 
    results = []
    emisss = []
    prev_x = trans_BOS.sample()
    prev_y = emiss[prev_x].sample()
    emisss.append(emiss[prev_x].probs)
    results.append(prev_y)
    for _ in range(1, steps):
        x = trans[prev_x].sample()
        y = emiss[x].sample()
        emisss.append(emiss[x].probs)
        results.append(y)
        prev_x = x
    return t.stack(results).unsqueeze(0), t.stack(emisss).unsqueeze(0)


m = CRF(3, batch_first = True)
m.opt = optim.AdamW(m.parameters(), 1e-3)

def train(m, times = 10):
    print(m.transitions)
    losss = []
    for i in range(times):
        res, emi = sample()
        log_prob = m(emi, res)
        loss = -log_prob
        losss.append(loss.item())
        m.zero_grad()
        loss.backward()
        m.opt.step()
    print(m.transitions)
    return losss

    

