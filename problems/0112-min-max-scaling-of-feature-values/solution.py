import torch

def min_max(x: torch.Tensor) -> torch.Tensor:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A tensor of numerical values
    
    Returns:
        A new tensor with values normalized to [0, 1]
    """
    x = torch.as_tensor(x, dtype=torch.float32)  # works for lists and int tensors too
    
    if x.numel() == 0:
        return x.clone()
    
    x_min = x.min()
    x_max = x.max()
    range_ = x_max - x_min
    
    if range_ == 0:
        return torch.zeros_like(x)
    
    return (x - x_min) / range_