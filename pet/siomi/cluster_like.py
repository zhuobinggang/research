from main import create_model
from reader import read_data

def train_model(data_points = 256, batch_size = 4):
    m = create_model()
    ds = read_data()
    assert data_points < 500
    train_ds = ds[:data_points]
    test_ds = ds[500:700]
    print('Training...')
    run_full(m, train_ds, batch_size = batch_size)
    print('Testing...')
    results, targets = get_test_result(m, test_ds)
    f = f1_score(targets, results, average='macro')
    prec = precision_score(targets, results, average='macro')
    rec = recall_score(targets, results, average='macro')
    print(f'Results: f {f}, precision {prec}, recall {rec}.')
    return m


