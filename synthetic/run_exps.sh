#python main.py mode=overfit dataset.biased=False
#python main.py mode=overfit dataset.biased=True
python main.py mode=prune dataset.biased=True prune.random_prune=False
#python main.py mode=prune dataset.biased=True prune.random_prune=True
#python main.py mode=prune dataset.biased=False prune.random_prune=False
#python main.py mode=prune dataset.biased=False prune.random_prune=True