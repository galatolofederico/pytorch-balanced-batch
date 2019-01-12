import torch
import torchvision
import torch.utils.data

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        idx = 0
        counter = {}
        sampled = 0
        while sampled < len(self.dataset):
            label = self._get_label(idx)
            if label not in counter:
                counter[label] = 0
            
            # always sample the class with fewer samples sampled
            min_label = min(counter.items(), key=lambda x: x[1])[0]

            if label == min_label:
                counter[label] += 1
                sampled += 1
                yield idx

            idx = (idx + 1) % len(self.dataset)
    
    def _get_label(self, idx):
        dataset_type = type(self.dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return self.dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return self.dataset.imgs[idx][1]
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.dataset)