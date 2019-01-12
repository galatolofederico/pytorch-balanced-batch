# pytorch-balanced-batch

A pytorch dataset ``sampler`` for always sampling balanced batches.

Be sure to use a ``batch_size`` that is an **integer multiple** of the **number of classes**.

For example, if your ``train_dataset`` has **10 classes** and you use a ``batch_size=30`` with the ``BalancedBatchSampler``

```python
train_loader = torch.utils.data.DataLoader(train_dataset, sampler=BalancedBatchSampler(train_dataset), batch_size=30)
```

You will obtain a ``train_loader`` in which each element has **3 samples** for each of the **10 classes**