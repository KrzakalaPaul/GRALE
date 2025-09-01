import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import torch
import numpy as np

model = nn.Linear(10,10)
warmup_steps = 1000
total_steps = 10000
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  

def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    # cosine annealing after warmup
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + np.cos(np.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda)

import matplotlib.pyplot as plt
lrs = []
for step in range(total_steps):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()
    
plt.plot(lrs)
plt.show()