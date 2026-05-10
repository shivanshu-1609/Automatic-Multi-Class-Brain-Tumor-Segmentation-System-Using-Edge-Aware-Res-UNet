import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def get_optimizer(model, base_lr=0.01, momentum=0.9, weight_decay=1e-4):
    """
    Creates and returns the SGD optimizer for the model.
    """
    return optim.SGD(
        model.parameters(), 
        lr=base_lr,
        momentum=momentum, 
        weight_decay=weight_decay
    )

def get_scheduler(optimizer, max_epochs, eta_min=1e-5, warmup_epochs=5):
    """
    Creates and returns a SequentialLR scheduler:
    1. Linear Warmup (epochs 0 to warmup_epochs)
    2. Cosine Annealing (epochs warmup_epochs to max_epochs)
    """
    # If max_epochs is very small, disable warmup
    if max_epochs <= warmup_epochs:
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=eta_min)
        
    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.01, # Starts at 1% of base_lr
        total_iters=warmup_epochs
    )
    
    cosine_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=(max_epochs - warmup_epochs), 
        eta_min=eta_min
    )
    
    return lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
