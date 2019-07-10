import torch
from sampler import BalancedBatchSampler

epochs = 3
size = 20
features = 5
classes_prob = torch.tensor([0.1, 0.4, 0.5])

dataset_X = torch.randn(size, features)
dataset_Y = torch.distributions.categorical.Categorical(classes_prob.repeat(size, 1)).sample()

dataset = torch.utils.data.TensorDataset(dataset_X, dataset_Y)

train_loader = torch.utils.data.DataLoader(dataset, sampler=BalancedBatchSampler(dataset, dataset_Y), batch_size=6)

for epoch in range(0, epochs):
    for batch_x, batch_y in train_loader:
        print("epoch: %d labels: %s\ninputs: %s\n" % (epoch, batch_y, batch_x))