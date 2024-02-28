## Setup variables
`time = torch.arange(0,20).float(); time`
`speed = torch.randn(20)*3 + 0.75*(time-9.5)**2 + 1`
```
def mse(preds, targets): return ((preds-targets)**2).mean()
```
## Step 1: Initialize the parameters
initialize the parameters to random values, and tell PyTorch that we want to track their gradients, using `requires_grad_`:
```
params = torch.randn(3).requires_grad_()
```
take a copy of `params`
```
orig_params = params.clone()
```
## Step 2: Calculate the predictions
```
preds = f(time, params)
```
## Step 3: Calculate the loss
```
loss = mse(preds, speed)
```
## Step 4: Calculate the gradients
```
loss.backward()
params.grad
```
Mulitply by 1e-5
```
params.grad * 1e-5
```
## Step 5: Step the weights.
```
lr = 1e-5
params.data -= lr * params.grad.data
params.grad = None
```

