import torch
import torch.nn as nn
from torch.nn import functional as F

class GPT(nn.Module):
    """ GPT Language Model """


    # ===============================================
    # This is the modified generate function in Q2.3
    # ===============================================

    @torch.no_grad()
    def generate(self, idx, device, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        input_prob = torch.ones(idx.shape).float().to(device)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            idx_cond.to(device)
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # print(probs.shape)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            select_prob = torch.gather(probs, 1, idx_next)
            input_prob = torch.cat((input_prob, select_prob), dim=1)
        return idx, input_prob


    # ===============================================
    # This is the modified generate function in Q2.4
    # ===============================================
    
    
    @torch.no_grad()
    def generate(self, idx, device, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        all_idx = torch.empty(1, 6).to(device)
        all_probs = torch.empty(1, 6).to(device)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            idx_cond.to(device)
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # print(probs.shape)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=6, dim=-1)
            # append sampled index to the running sequence and continue
            # print(idx.shape, idx_next.shape)
            idx = torch.cat((idx, idx_next[0][0].reshape(1, 1)), dim=1)
            select_prob = torch.gather(probs, 1, idx_next)
            all_probs = torch.cat((all_probs, select_prob), dim=0)
            all_idx = torch.cat((all_idx, idx_next), dim=0)
        return idx, all_idx[1:], all_probs[1:]
