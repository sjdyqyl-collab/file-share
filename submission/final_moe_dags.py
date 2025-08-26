import graphviz

# Create the final accurate baseline DAG
def create_final_baseline_dag():
    dot = graphviz.Digraph('baseline_moe_final', format='svg')
    dot.attr(rankdir='TB', size='35,45', ranksep='1.0', nodesep='0.5')
    
    # Input
    dot.node('input', 'Input\n[1024 tokens]\nAll 16 GPUs', 
             shape='ellipse', fillcolor='lightcoral')
    
    # Layer 1 - 16 experts, 4 per GPU across 4 GPUs, but with TP=8, PP=2
    # This means: 16 GPUs total, divided into 2 pipeline stages of 8 GPUs each
    # 16 experts / 4 experts per GPU = 4 GPUs needed, but we have 16 GPUs
    # Actually: 16 experts total, 4 experts/GPU means 4 GPUs used for experts
    # But with TP=8, PP=2, we have 16 GPUs total
    
    # Let's clarify: 16 experts per layer, 4 experts per GPU = 4 GPUs for experts
    # But baseline uses 16 GPUs with TP=8, PP=2, so this is confusing
    # The paper says: baseline has 16 GPUs, TP=8, PP=2, 4 experts/GPU
    # This suggests: 16 experts total / 4 experts per GPU = 4 GPUs used
    # But 16 GPUs with TP=8, PP=2 means 2 pipeline stages of 8 GPUs each
    # So experts are distributed across the 16 GPUs
    
    for layer in range(1, 5):
        with dot.subgraph(name=f'layer{layer}') as c:
            c.attr(label=f'Layer {layer}')
            
            # Multi-head attention (TP=8 across GPUs)
            c.node(f'l{layer}_attn', f'MHA\n[1024, hidden]\nTP8 GPUs 0-7,8-15', 
                   fillcolor='lightblue')
            
            # Expert routing
            c.node(f'l{layer}_gate', f'Gate\n[1024 tokens]\nAll 16 GPUs', 
                   shape='parallelogram')
            
            # Expert computation - 16 experts, 4 per GPU
            for gpu in range(16):
                for exp in range(4):
                    expert_id = gpu * 4 + exp
                    if expert_id < 16:  # Only 16 experts total
                        c.node(f'l{layer}_gpu{gpu}_exp{exp}', 
                              f'Expert {expert_id}\nGPU {gpu}\n[Variable tokens]',
                              fillcolor='lightblue')
            
            # Expert aggregation
            c.node(f'l{layer}_agg', f'Aggregate\n[1024 tokens]\nAll 16 GPUs', 
                   shape='parallelogram')
            
            # Residual
            c.node(f'l{layer}_res', f'Residual\n[1024 tokens]\nAll 16 GPUs', 
                   shape='diamond')
    
    # Output
    dot.node('output', 'Output\n[1024 tokens]\nGPU 15', 
             shape='ellipse', fillcolor='lightcoral')
    
    # Connections
    dot.edge('input', 'l1_attn')
    dot.edge('l1_attn', 'l1_gate')
    
    # Connect gate to experts
    for gpu in range(16):
        for exp in range(4):
            expert_id = gpu * 4 + exp
            if expert_id < 16:
                dot.edge('l1_gate', f'l1_gpu{gpu}_exp{exp}', style='dashed')
    
    # Connect experts to aggregation
    for gpu in range(16):
        for exp in range(4):
            expert_id = gpu * 4 + exp
            if expert_id < 16:
                dot.edge(f'l1_gpu{gpu}_exp{exp}', 'l1_agg')
    
    dot.edge('l1_agg', 'l1_res')
    dot.edge('l1_res', 'l2_attn')
    dot.edge('l2_attn', 'l2_gate')
    
    # Continue pattern...
    for layer in range(2, 5):
        for gpu in range(16):
            for exp in range(4):
                expert_id = gpu * 4 + exp
                if expert_id < 16:
                    dot.edge(f'l{layer}_gate', f'l{layer}_gpu{gpu}_exp{exp}', style='dashed')
                    dot.edge(f'l{layer}_gpu{gpu}_exp{exp}', f'l{layer}_agg')
        
        dot.edge(f'l{layer}_agg', f'l{layer}_res')
        if layer < 4:
            dot.edge(f'l{layer}_res', f'l{layer+1}_attn')
    
    dot.edge('l4_res', 'output')
    
    return dot

# Create the final accurate proposed DAG
def create_final_proposed_dag():
    dot = graphviz.Digraph('proposed_moe_final', format='svg')
    dot.attr(rankdir='TB', size='50,60', ranksep='1.0', nodesep='0.4')
    
    # Input
    dot.node('input', 'Input\n[1024 tokens, hidden_dim]\nBroadcast to 64 GPUs', 
             shape='ellipse', fillcolor='lightcoral')
    
    # 4 layers, each with 16 experts on separate GPUs
    for layer in range(4):
        layer_num = layer + 1
        gpu_base = layer * 16
        
        with dot.subgraph(name=f'layer{layer_num}') as c:
            c.attr(label=f'Layer {layer_num} (Experts {layer*16}-{(layer+1)*16-1} on GPUs {gpu_base}-{gpu_base+15})')
            
            # Multi-head attention (replicated on all 16 GPUs for this layer)
            c.node(f'l{layer_num}_attn', 
                  f'Multi-Head Attention\n[1024 tokens, hidden_dim]\nReplicated on GPUs {gpu_base}-{gpu_base+15}',
                  fillcolor='lightblue')
            
            # Gate computation
            c.node(f'l{layer_num}_gate', 
                  f'Gate Network\n[1024 tokens]\nCompute routing probabilities\nGPUs {gpu_base}-{gpu_base+15}',
                  shape='parallelogram')
            
            # Token splitting based on routing
            c.node(f'l{layer_num}_split', 
                  f'Async Token Split\n[Variable tokens per expert]\nSend to target GPUs',
                  shape='parallelogram')
            
            # 16 experts, each on separate GPU
            for expert in range(16):
                expert_id = layer * 16 + expert
                gpu_id = gpu_base + expert
                
                c.node(f'l{layer_num}_expert{expert}_gpu{gpu_id}', 
                      f'Expert {expert_id}\nGPU {gpu_id}\n[Variable tokens, expert_dim]\n1 expert/GPU',
                      fillcolor='lightblue')
            
            # Gather results from all experts
            c.node(f'l{layer_num}_gather', 
                  f'Async All-Gather\n[1024 tokens]\nCollect from GPUs {gpu_base}-{gpu_base+15}',
                  shape='parallelogram')
            
            # Residual connection
            c.node(f'l{layer_num}_residual', 
                  f'Residual Add\n[1024 tokens]\nAcross GPUs {gpu_base}-{gpu_base+15}',
                  shape='diamond')
    
    # Output
    dot.node('output', 'Output\n[1024 tokens, hidden_dim]\nFrom GPU 63', 
             shape='ellipse', fillcolor='lightcoral')
    
    # Detailed connections
    dot.edge('input', 'l1_attn')
    dot.edge('l1_attn', 'l1_gate')
    dot.edge('l1_gate', 'l1_split')
    
    # Connect split to experts with routing
    for layer in range(4):
        layer_num = layer + 1
        gpu_base = layer * 16
        
        for expert in range(16):
            expert_id = layer * 16 + expert
            gpu_id = gpu_base + expert
            dot.edge(f'l{layer_num}_split', f'l{layer_num}_expert{expert}_gpu{gpu_id}', 
                    style='dashed', label=f'tokens routed to GPU {gpu_id}')
        
        # Connect experts to gather
        for expert in range(16):
            expert_id = layer * 16 + expert
            gpu_id = gpu_base + expert
            dot.edge(f'l{layer_num}_expert{expert}_gpu{gpu_id}', f'l{layer_num}_gather')
        
        dot.edge(f'l{layer_num}_gather', f'l{layer_num}_residual')
        
        # Connect to next layer
        if layer < 3:
            dot.edge(f'l{layer_num}_residual', f'l{layer+2}_attn')
        else:
            dot.edge('l4_residual', 'output')
    
    return dot

# Create communication-focused DAGs
if __name__ == "__main__":
    # Create final baseline
    baseline_final = create_final_baseline_dag()
    baseline_final.render('/home/wzc/data/file-share/submission/baseline_moe_final')
    
    # Create final proposed
    proposed_final = create_final_proposed_dag()
    proposed_final.render('/home/wzc/data/file-share/submission/proposed_moe_final')
    
    # Create communication pattern DAG for proposed
    comm_dot = graphviz.Digraph('proposed_communication', format='svg')
    comm_dot.attr(rankdir='LR', size='30,20')
    
    # Show communication pattern
    comm_dot.attr('node', shape='ellipse', fillcolor='lightgreen')
    
    # Input broadcast
    comm_dot.node('input_broadcast', 'Input Broadcast\n[1024 tokens]\nTo 64 GPUs')
    
    # Layer communications
    for layer in range(4):
        comm_dot.node(f'layer{layer+1}_routing', f'Layer {layer+1} Routing\nAsync token routing\nTo 16 experts')
        comm_dot.node(f'layer{layer+1}_gather', f'Layer {layer+1} Gather\nAsync all-gather\nFrom 16 experts')
    
    # Output
    comm_dot.node('output_collect', 'Output Collection\n[1024 tokens]\nFrom GPU 63')
    
    # Connections
    comm_dot.edge('input_broadcast', 'layer1_routing')
    comm_dot.edge('layer1_routing', 'layer1_gather')
    comm_dot.edge('layer1_gather', 'layer2_routing')
    comm_dot.edge('layer2_routing', 'layer2_gather')
    comm_dot.edge('layer2_gather', 'layer3_routing')
    comm_dot.edge('layer3_routing', 'layer3_gather')
    comm_dot.edge('layer3_gather', 'layer4_routing')
    comm_dot.edge('layer4_routing', 'layer4_gather')
    comm_dot.edge('layer4_gather', 'output_collect')
    
    comm_dot.render('/home/wzc/data/file-share/submission/proposed_communication')
    
    print("Final DAGs generated successfully!")