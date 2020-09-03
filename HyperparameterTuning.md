# Hyperparameter Tuning with ray[tune]

## Why Hyperparameter Tuning?

Modern ML models are based on neural networks and highly dependent on hyperparameters. But also standard ml methods like Regression can benefit from automatic tuning pipelines.
Also hyperparameters can change with new data, data size and so on.

A typical training loop can be written like:
```python
def train_model():
    model = ConvNet()
    optimizer = Optimizer()
    for batch in Dataset():
        loss, accuracy = model.train(batch)
        optimizer.update(model, loss)
```
But training costs money and time! And in the background there are a lot of parameters to set and tune:

```python
def train_model():
    model = ConvNet(layers, activations, ....)
    optimizer = Optimizer(learningrate, momentum, decay,...)
    for batch in Dataset(standardize, shift, ...):
        loss, accuracy = model.train(batch)
        optimizer.update(model, loss)
```

![Model Size influence on performance](assets/halfcheetah-v1-model-size.png)
