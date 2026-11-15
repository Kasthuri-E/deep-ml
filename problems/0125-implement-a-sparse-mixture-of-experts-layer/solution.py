import torch

def moe(x, We, Wg, n_experts, top_k):
    x = torch.as_tensor(x, dtype=torch.float64)
    We = torch.as_tensor(We, dtype=torch.float64)
    Wg = torch.as_tensor(Wg, dtype=torch.float64)

    n_batch, l_seq, d_model = x.shape
    tokens = x.reshape(-1, d_model)                         # (T, d_model)

    # Gating: softmax over all experts
    probs = torch.softmax(tokens @ Wg, dim=-1)              # (T, n_experts)

    # Top-k selection and renormalization
    top_p, top_idx = torch.topk(probs, top_k, dim=-1)       # (T, k)
    top_p = top_p / top_p.sum(dim=-1, keepdim=True)

    # All expert outputs: (T, n_experts, d_out)
    expert_out = torch.einsum("td,edo->teo", tokens, We)
    selected = torch.gather(
        expert_out, 1,
        top_idx.unsqueeze(-1).expand(-1, -1, expert_out.size(-1))
    )                                                       # (T, k, d_out)

    out = (selected * top_p.unsqueeze(-1)).sum(dim=1)       # (T, d_out)
    return out.reshape(n_batch, l_seq, -1)