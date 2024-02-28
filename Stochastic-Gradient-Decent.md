## Setup variables
`time = torch.arange(0,20).float(); time`
`speed = torch.randn(20)*3 + 0.75*(time-9.5)**2 + 1`

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
