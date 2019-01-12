import torch
from torchvision import datasets, transforms
from sampler import BalancedBatchSampler

train_dataset = datasets.MNIST("/tmp/mnist_data",
                train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        )

train_loader = torch.utils.data.DataLoader(train_dataset, sampler=BalancedBatchSampler(train_dataset), batch_size=30)

for batch_x, batch_y in train_loader:
    print(batch_y)