import torch

def get_neighbors(agent_positions, all_agent_positions, radius=2.0):
    """
    Finds nearby agents at each timestep.
    
    Args:
        agent_positions: tensor of shape [4, 2] (current agent's (x,y) track)
        all_agent_positions: list of tensors, each [4, 2] (other agents in scene)
        radius: float, distance threshold in meters
        
    Returns:
        List of length 4. Each element is a list of [2] tensors representing 
        neighbor (x,y) positions at that specific timestep.
    """
    seq_len = agent_positions.shape[0]
    neighbors_over_time = [[] for _ in range(seq_len)]
    
    for t in range(seq_len):
        current_pos = agent_positions[t]
        
        for other_agent in all_agent_positions:
            other_pos = other_agent[t]
            distance = torch.norm(current_pos - other_pos)
            
            if distance <= radius:
                neighbors_over_time[t].append(other_pos)
                
    return neighbors_over_time  