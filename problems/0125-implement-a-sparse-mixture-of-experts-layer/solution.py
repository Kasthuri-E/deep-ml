import numpy as np

def moe(x, We, Wg, n_experts, top_k):
    x = np.asarray(x, dtype=float)
    We = np.asarray(We, dtype=float)
    Wg = np.asarray(Wg, dtype=float)

    n_batch, l_seq, d_model = x.shape
    tokens = x.reshape(-1, d_model)                     # (T, d_model)

    # Gating: softmax over all experts (numerically stable)
    logits = tokens @ Wg                                # (T, n_experts)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)

    # Top-k selection (stable sort -> ties go to lower expert index)
    top_idx = np.argsort(-probs, axis=1, kind="stable")[:, :top_k]   # (T, k)
    top_p = np.take_along_axis(probs, top_idx, axis=1)               # (T, k)
    top_p /= top_p.sum(axis=1, keepdims=True)                        # renormalize

    # All expert outputs: (T, n_experts, d_out)
    expert_out = np.einsum("td,edo->teo", tokens, We)
    selected = np.take_along_axis(expert_out, top_idx[:, :, None], axis=1)  # (T, k, d_out)

    out = (selected * top_p[:, :, None]).sum(axis=1)                 # (T, d_out)
    return out.reshape(n_batch, l_seq, -1)