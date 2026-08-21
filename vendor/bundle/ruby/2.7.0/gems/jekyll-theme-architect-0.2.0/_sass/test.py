# GRPO Python Code
import torch

def compute_grpo_advantage(rewards):
    group_mean = rewards.mean(dim=1)
    
    group_std = rewards.std()
    
    advantages = (rewards - group_mean) / group_std
    return advantages



def grpo_train_step(model, ref_model, optimizer, new_log_probs, old_log_probs, rewards, clip_eps, response_mask):
    
    advantages = compute_grpo_advantage(rewards)
    ratio = torch.exp(new_log_probs-old_log_probs)
    obj = ratio * advantages
    
    clip_obj = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
    
    final_obj = torch.minimum(obj, clip_obj)
    
    kl = None
    
    final_obj -= kl
    final_loss = -final_obj * response_mask
    final_loss.backward()
    optimizer.step()
    return final_loss
    
    
        