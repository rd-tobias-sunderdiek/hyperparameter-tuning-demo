# Hyperparameter Tuning with ray[tune]

## Why Hyperparameter Tuning?

Modern ML models are based on neural networks and highly dependent on hyperparameters. But also standard ml methods like Regression can benefit from automatic tuning pipelines.
Also hyperparameters can change with new data, data size and so on.

A typical training loops can be written like:
```python
def train_model():
    model = ConvNet()
    optimizer = Optimizer()
    for batch in Dataset():
        loss, accuracy = model.train(batch)
        optimizer.update(model, loss)
```

