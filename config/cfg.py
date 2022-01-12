cfg = {
    'gpu_id': 0,
    'max_len': 120,
    'train_batch_size': 16 * 5,
    'test_batch_size': 16 * 5,
    'learning_rate': 1e-5,
    'epoch': 8,
    'K': 16,
    'Kt': 50,
    # 'template': '[X1] [X2]? [MASK].',
    'template': '[X1] ? [MASK] , [X2]',
    'answer': ['No', 'Yes'],
    'device': 'TPU',
    'optimizer': 'Adam',
    'word_size': 50265
}

hyper_roberta = {
    'word_dim': 1024,
    'dropout': 0.1
}

path = {
    'train_path': 'data/SST5/sst_train.txt',
    'test_path': 'data/SST5/sst_test.txt',
    'roberta_path': 'roberta-large'
}
