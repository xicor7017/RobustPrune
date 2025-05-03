import time
import torch
import random
from copy import deepcopy
from torch.distributions import Normal

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def get_dataloaders(num_training, num_testing, num_miss, batch_size, all_train_batch_size=-1, biased=False):
    random.seed(1)
    torch.manual_seed(1)

    data = Data(num_training, num_testing, num_miss, biased=biased)

    training_data = data.training_data
    training_labels = data.training_labels
    training_dataset = CustomDataset(training_data, training_labels)

    testing_data = data.testing_data
    testing_labels = data.testing_labels
    testing_dataset = CustomDataset(testing_data, testing_labels)

    miss_data = data.miss_data
    miss_labels = data.miss_labels
    miss_dataset = CustomDataset(miss_data, miss_labels)

    all_train_dataset = CustomDataset(torch.cat([training_data, miss_data]), torch.cat([training_labels, miss_labels]))

    if batch_size == -1:
        training_batch_size = len(training_dataset)
        testing_batch_size  = len(testing_dataset)
        miss_batch_size     = len(miss_dataset)
    else:
        training_batch_size = batch_size
        testing_batch_size  = batch_size
        miss_batch_size     = batch_size


    if all_train_batch_size == -1:
        all_train_batch_size = len(all_train_dataset)
    else:
        all_train_batch_size = all_train_batch_size

    train_dataloader = DataLoader(training_dataset, training_batch_size, shuffle=True)
    test_dataloader  = DataLoader(testing_dataset, testing_batch_size, shuffle=True)
    miss_dataloader  = DataLoader(miss_dataset, miss_batch_size, shuffle=True)
    all_train_dataloader = DataLoader(all_train_dataset, all_train_batch_size, shuffle=True)

    grid_data = data.get_spread()

    return train_dataloader, test_dataloader, miss_dataloader, all_train_dataloader, grid_data.view((-1, 2))

class CustomDataset(Dataset):
    def __init__(self, datapoints, labels):
        self.labels     = labels
        self.datapoints = datapoints

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        return self.datapoints[idx], self.labels[idx].long()

class Data:
    def __init__(self, num_training=100, num_testing=10, num_miss=5, biased=False):
        '''
        num_training    : number of training samples
        num_testing     : number of testing samples
        num_miss        : number of missing_labeled samples
        biased          : if True, the missing samples are all come from the same class
        '''

        self.biased = biased
        self.means = torch.FloatTensor([ 
                                            [0, 0],
                                            [1, 1],
                                            [1, -1],
                                            [-1, 1],
                                            [-1, -1],
                                                        ])

        self.std = torch.FloatTensor([0.20, 0.20])      # x,y stds
        labels   = [0,1,2,3,4]                          # Class labels

        self.distributions      = [Normal(mean, self.std) for mean in self.means]
        self.miss_distributions = [Normal(mean, self.std) for mean in self.means]

        self.testing_data  = torch.stack([distribution.sample((num_testing,)) for distribution in self.distributions]).reshape((-1, 2))
        self.training_data = torch.stack([distribution.sample((num_training,)) for distribution in self.distributions]).reshape((-1, 2))

        self.testing_labels  = torch.FloatTensor(labels).unsqueeze(-1).repeat((1, num_testing)).reshape(-1)
        self.training_labels = torch.FloatTensor(labels).unsqueeze(-1).repeat((1, num_training)).reshape(-1)

        while True:
            shuffled_indices   = list(range(len(self.means)))
            unshuffled_indices = torch.LongTensor(deepcopy(shuffled_indices))
            random.shuffle(shuffled_indices)
            shuffled_indices = torch.LongTensor(shuffled_indices)
            if not (unshuffled_indices == shuffled_indices).any():
                break
        
        if self.biased:
            self.miss_data   = torch.stack([self.miss_distributions[i].sample((num_miss,)) for i in shuffled_indices]).reshape((-1, 2))
            self.miss_labels = torch.FloatTensor(labels).unsqueeze(-1).repeat((1, num_miss)).reshape(-1)
        else:
            self.miss_data = []
            self.miss_labels = []
            all_labels = [0,1,2,3,4]
            for i, miss_distribution in enumerate(self.miss_distributions):
                # Remove i from all_labels
                labels = all_labels[:i] + all_labels[i+1:]
                for _ in range(num_miss):
                    sample = miss_distribution.sample()
                    self.miss_data.append(sample)
                    random_label = random.choice(labels)
                    self.miss_labels.append(random_label)
            
            self.miss_data = torch.stack(self.miss_data)
            self.miss_labels = torch.FloatTensor(self.miss_labels)
        
    def all_different(self, shuffled_labels, labels):
        for l, sl in zip(labels, shuffled_labels):
            if l == sl:
                return False
        return True

    def show(self):
        #Plot training data and testing data
        import matplotlib.pyplot as plt
        plt.scatter(self.training_data[:,0], self.training_data[:,1], c=self.training_labels)
        plt.scatter(self.miss_data[:,0], self.miss_data[:,1], c=self.miss_labels)
        plt.savefig("synthetic.png")

    def get_spread(self):
        min_x = torch.min(self.training_data[:,0]) - 3*self.std[0]
        max_x = torch.max(self.training_data[:,0]) + 3*self.std[0]
        min_y = torch.min(self.training_data[:,1]) - 3*self.std[1]
        max_y = torch.max(self.training_data[:,1]) + 3*self.std[1]
        # generate 250*250 grid
        x = torch.linspace(min_x, max_x, 250)
        y = torch.linspace(min_y, max_y, 250)
        xx, yy = torch.meshgrid(x, y)
        grid = torch.stack([xx, yy], dim=-1)
        return grid

if __name__ == "__main__":
    data = Data(biased=False)
    data.show()