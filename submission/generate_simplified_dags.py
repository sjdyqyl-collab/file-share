import graphviz

def create_dense_ring_attention_dag():
    """Create a complete DAG for Ring Attention + Sequence Parallelism (Dense Model)"""
    
    dot = graphviz.Digraph(comment='Ring Attention + Sequence Parallelism - Dense Transformer')
    dot.attr(rankdir='TB', size='20,20', ranksep='1.0')
    
    # Define colors for different GPU nodes
    colors = {
        0: 'lightblue',
        1: 'lightgreen', 
        2: 'lightyellow',
        3: 'lightcoral'
    }
    
    num_gpus = 4
    
    # Create input layer
    dot.node('total_input', 'Total Input\n(B, L, d_model)', shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Create GPU clusters for sequence parallelism
    for gpu_id in range(num_gpus):
        with dot.subgraph(name=f'cluster_gpu_{gpu_id}') as c:
            c.attr(label=f'GPU {gpu_id} (Sequence Parallel)', style='rounded', bgcolor=colors[gpu_id], color='black')
            
            # Input split
            c.node(f'input_{gpu_id}', f'Sequence Split\nX[{gpu_id}]\n(B, L/4, d_model)', 
                   shape='ellipse', style='filled', fillcolor='white')
            
            # Q,K,V projections
            c.node(f'qkv_{gpu_id}', f'QKV Projections\n(B, L/4, d_model)', 
                   shape='rectangle', style='filled', fillcolor='white')
            
            # Ring attention stages
            for stage in range(num_gpus):
                src_idx = (gpu_id - stage) % num_gpus
                c.node(f'ring_stage_{gpu_id}_{stage}', 
                       f'Ring Stage {stage}\nAttn(Q[{gpu_id}], K[{src_idx}], V[{src_idx}])\n(B, L/4, d_model)', 
                       shape='rectangle', style='filled,rounded', fillcolor='white')
            
            # Accumulation
            c.node(f'accum_{gpu_id}', f'Accumulate\nRing Results\n(B, L/4, d_model)', 
                   shape='ellipse', style='filled', fillcolor='white')
            
            # Output projection
            c.node(f'out_proj_{gpu_id}', f'Output Projection\n(B, L/4, d_model)', 
                   shape='rectangle', style='filled', fillcolor='white')
            
            # Residual
            c.node(f'residual_{gpu_id}', f'Residual Add\n(B, L/4, d_model)', 
                   shape='ellipse', style='filled', fillcolor='white')
            
            # Final output
            c.node(f'output_{gpu_id}', f'Output Chunk\n(B, L/4, d_model)', 
                   shape='ellipse', style='filled', fillcolor='white')
    
    # Communication nodes for ring topology
    for stage in range(num_gpus):
        for gpu_id in range(num_gpus):
            next_gpu = (gpu_id + 1) % num_gpus
            dot.node(f'comm_{gpu_id}_{stage}', 
                    f'KV Transfer\nGPU {gpu_id}→{next_gpu}\nStage {stage}\n(B, L/4, d_model)', 
                    shape='diamond', style='filled', fillcolor='lightgray')
    
    # Connections
    # Input splitting
    for gpu_id in range(num_gpus):
        dot.edge('total_input', f'input_{gpu_id}')
        dot.edge(f'input_{gpu_id}', f'qkv_{gpu_id}')
    
    # Ring attention flow
    for gpu_id in range(num_gpus):
        dot.edge(f'qkv_{gpu_id}', f'ring_stage_{gpu_id}_0')
        
        for stage in range(1, num_gpus):
            # Communication before each stage
            prev_gpu = (gpu_id - 1) % num_gpus
            dot.edge(f'comm_{prev_gpu}_{stage-1}', f'ring_stage_{gpu_id}_{stage}')
            dot.edge(f'ring_stage_{gpu_id}_{stage-1}', f'comm_{gpu_id}_{stage-1}')
        
        # Accumulation chain
        dot.edge(f'ring_stage_{gpu_id}_0', f'accum_{gpu_id}')
        for stage in range(1, num_gpus):
            dot.edge(f'ring_stage_{gpu_id}_{stage}', f'accum_{gpu_id}')
        
        # Output processing
        dot.edge(f'accum_{gpu_id}', f'out_proj_{gpu_id}')
        dot.edge(f'out_proj_{gpu_id}', f'residual_{gpu_id}')
        dot.edge(f'input_{gpu_id}', f'residual_{gpu_id}')  # Residual connection
        dot.edge(f'residual_{gpu_id}', f'output_{gpu_id}')
    
    # Final aggregation
    dot.node('final_output', 'Final Output\n(B, L, d_model)', shape='ellipse', style='filled', fillcolor='lightgray')
    for gpu_id in range(num_gpus):
        dot.edge(f'output_{gpu_id}', 'final_output')
    
    return dot

def create_moe_simplified_dag():
    """Create a simplified MoE DAG focusing on key concepts"""
    
    dot = graphviz.Digraph(comment='MoE Ring Attention + Sequence Parallelism - Simplified')
    dot.attr(rankdir='TB', size='20,20', ranksep='1.2')
    
    colors = {
        0: 'lightblue',
        1: 'lightgreen', 
        2: 'lightyellow',
        3: 'lightcoral'
    }
    
    num_gpus = 4
    
    # Input
    dot.node('total_input', 'Total Input\n(B, L, d_model)', shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Create GPU clusters
    for gpu_id in range(num_gpus):
        with dot.subgraph(name=f'cluster_gpu_{gpu_id}') as c:
            c.attr(label=f'GPU {gpu_id}', style='rounded', bgcolor=colors[gpu_id], color='black')
            
            # Input split
            c.node(f'input_{gpu_id}', f'Sequence Split\nX[{gpu_id}]\n(B, L/4, d_model)', 
                   shape='ellipse', style='filled', fillcolor='white')
            
            # Attention (ring attention simplified)
            c.node(f'attention_{gpu_id}', f'Ring Attention\n(B, L/4, d_model)', 
                   shape='rectangle', style='filled', fillcolor='white')
            
            # Layer norm
            c.node(f'ln1_{gpu_id}', f'LayerNorm\n(B, L/4, d_model)', 
                   shape='rectangle', style='filled', fillcolor='white')
            
            # Gate
            c.node(f'gate_{gpu_id}', f'Gate\n(B, L/4, num_experts)', 
                   shape='parallelogram', style='filled', fillcolor='lightpink')
            
            # Local experts (2 per GPU for 8 total)
            for local_expert in range(2):
                expert_id = gpu_id * 2 + local_expert
                c.node(f'expert_{gpu_id}_{expert_id}', 
                       f'Expert {expert_id}\nMLP\n(B, k, d_model)', 
                       shape='rectangle', style='filled', fillcolor='lightcyan')
            
            # Aggregation
            c.node(f'aggregate_{gpu_id}', f'Aggregate\nExpert Outputs\n(B, L/4, d_model)', 
                   shape='parallelogram', style='filled', fillcolor='white')
            
            # Final processing
            c.node(f'ln2_{gpu_id}', f'LayerNorm\n(B, L/4, d_model)', 
                   shape='rectangle', style='filled', fillcolor='white')
            c.node(f'residual_{gpu_id}', f'Residual Add\n(B, L/4, d_model)', 
                   shape='ellipse', style='filled', fillcolor='white')
            c.node(f'output_{gpu_id}', f'Output Chunk\n(B, L/4, d_model)', 
                   shape='ellipse', style='filled', fillcolor='white')
    
    # Expert communication
    for gpu_id in range(num_gpus):
        for target_gpu in range(num_gpus):
            if gpu_id != target_gpu:
                dot.node(f'expert_comm_{gpu_id}_{target_gpu}', 
                        f'Expert Routing\nGPU {gpu_id}→{target_gpu}\n(B, k, d_model)', 
                        shape='diamond', style='filled', fillcolor='lightgray')
    
    # Connections
    for gpu_id in range(num_gpus):
        dot.edge('total_input', f'input_{gpu_id}')
        dot.edge(f'input_{gpu_id}', f'attention_{gpu_id}')
        dot.edge(f'attention_{gpu_id}', f'ln1_{gpu_id}')
        dot.edge(f'ln1_{gpu_id}', f'gate_{gpu_id}')
        
        # Gate to experts (local and remote)
        for target_gpu in range(num_gpus):
            expert_id = target_gpu * 2  # First expert on each GPU
            if gpu_id == target_gpu:
                # Local expert
                dot.edge(f'gate_{gpu_id}', f'expert_{target_gpu}_{expert_id}', 
                        style='dashed', label='select')
                dot.edge(f'ln1_{gpu_id}', f'expert_{target_gpu}_{expert_id}')
                dot.edge(f'expert_{target_gpu}_{expert_id}', f'aggregate_{gpu_id}')
            else:
                # Remote expert
                dot.edge(f'gate_{gpu_id}', f'expert_comm_{gpu_id}_{target_gpu}', 
                        style='dashed', label='route')
                dot.edge(f'ln1_{gpu_id}', f'expert_comm_{gpu_id}_{target_gpu}')
                dot.edge(f'expert_comm_{gpu_id}_{target_gpu}', f'expert_{target_gpu}_{expert_id}')
                dot.edge(f'expert_{target_gpu}_{expert_id}', f'aggregate_{gpu_id}')
        
        # Second expert on same GPU
        expert_id = gpu_id * 2 + 1
        dot.edge(f'gate_{gpu_id}', f'expert_{gpu_id}_{expert_id}', 
                style='dashed', label='select')
        dot.edge(f'ln1_{gpu_id}', f'expert_{gpu_id}_{expert_id}')
        dot.edge(f'expert_{gpu_id}_{expert_id}', f'aggregate_{gpu_id}')
        
        # Output processing
        dot.edge(f'aggregate_{gpu_id}', f'ln2_{gpu_id}')
        dot.edge(f'ln2_{gpu_id}', f'residual_{gpu_id}')
        dot.edge(f'attention_{gpu_id}', f'residual_{gpu_id}')  # Residual
        dot.edge(f'residual_{gpu_id}', f'output_{gpu_id}')
    
    # Final aggregation
    dot.node('final_output', 'Final Output\n(B, L, d_model)', shape='ellipse', style='filled', fillcolor='lightgray')
    for gpu_id in range(num_gpus):
        dot.edge(f'output_{gpu_id}', 'final_output')
    
    return dot

# Generate both DAGs
if __name__ == "__main__":
    # Dense transformer DAG
    dense_dag = create_dense_ring_attention_dag()
    dense_dag.render('/home/wzc/data/file-share/submission/ring_attention_dense', format='svg', cleanup=True)
    
    # MoE transformer DAG  
    moe_dag = create_moe_simplified_dag()
    moe_dag.render('/home/wzc/data/file-share/submission/ring_attention_moe', format='svg', cleanup=True)
    
    print("Simplified DAGs generated successfully!")
    print("Dense model DAG: /home/wzc/data/file-share/submission/ring_attention_dense.svg")
    print("MoE model DAG: /home/wzc/data/file-share/submission/ring_attention_moe.svg")