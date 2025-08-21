import graphviz

def create_ring_attention_dag():
    """Create a complete DAG for Ring Attention + Sequence Parallelism deployment"""
    
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Ring Attention + Sequence Parallelism DAG')
    dot.attr(rankdir='TB', size='20,20')
    
    # Define colors for different GPU nodes
    colors = {
        0: 'lightblue',
        1: 'lightgreen', 
        2: 'lightyellow',
        3: 'lightcoral'
    }
    
    # Define number of GPUs for this example (4 GPUs as per paper's ring topology)
    num_gpus = 4
    
    # Input splitting across GPUs using sequence parallelism
    for gpu_id in range(num_gpus):
        with dot.subgraph(name=f'cluster_gpu_{gpu_id}') as c:
            c.attr(label=f'GPU {gpu_id}', style='rounded', bgcolor=colors[gpu_id])
            
            # Input processing
            c.node(f'input_{gpu_id}', f'Input Split\nX[{gpu_id}]\n(B, L/4, d_model)', 
                   shape='ellipse', style='filled', fillcolor='white')
            
            # Q,K,V projections
            c.node(f'q_proj_{gpu_id}', f'Q Projection\nX[{gpu_id}]W_Q\n(B, L/4, d_model)', 
                   shape='rectangle', style='filled', fillcolor='white')
            c.node(f'k_proj_{gpu_id}', f'K Projection\nX[{gpu_id}]W_K\n(B, L/4, d_model)', 
                   shape='rectangle', style='filled', fillcolor='white')
            c.node(f'v_proj_{gpu_id}', f'V Projection\nX[{gpu_id}]W_V\n(B, L/4, d_model)', 
                   shape='rectangle', style='filled', fillcolor='white')
            
            # Split into heads
            c.node(f'q_split_{gpu_id}', f'Split Heads\nQ[{gpu_id}]\n(B, L/4, H, d_h)', 
                   shape='parallelogram', style='filled', fillcolor='white')
            c.node(f'k_split_{gpu_id}', f'Split Heads\nK[{gpu_id}]\n(B, L/4, H, d_h)', 
                   shape='parallelogram', style='filled', fillcolor='white')
            c.node(f'v_split_{gpu_id}', f'Split Heads\nV[{gpu_id}]\n(B, L/4, H, d_h)', 
                   shape='parallelogram', style='filled', fillcolor='white')
    
    # Ring attention stages
    for stage in range(num_gpus):
        with dot.subgraph(name=f'cluster_stage_{stage}') as c:
            c.attr(label=f'Ring Stage {stage}', style='dashed')
            
            for gpu_id in range(num_gpus):
                src_idx = (gpu_id - stage) % num_gpus
                
                # Attention computation for this stage
                dot.node(f'attn_{gpu_id}_{stage}', 
                        f'Attention\nStage {stage}\nGPU {gpu_id}\nQ[{gpu_id}]×K[{src_idx}]×V[{src_idx}]\n(B, L/4, H, d_h)', 
                        shape='rectangle', style='filled', fillcolor=colors[gpu_id])
                
                # Accumulation
                if stage == 0:
                    dot.node(f'accum_{gpu_id}_{stage}', 
                            f'Initialize\nGPU {gpu_id}\n(B, L/4, H, d_h)', 
                            shape='ellipse', style='filled', fillcolor=colors[gpu_id])
                else:
                    dot.node(f'accum_{gpu_id}_{stage}', 
                            f'Accumulate\nGPU {gpu_id}\nAdd partial result\n(B, L/4, H, d_h)', 
                            shape='ellipse', style='filled', fillcolor=colors[gpu_id])
    
    # Communication paths for ring topology
    for stage in range(num_gpus):
        for gpu_id in range(num_gpus):
            next_gpu = (gpu_id + 1) % num_gpus
            
            # KV transfer in ring
            if stage < num_gpus - 1:
                dot.node(f'comm_{gpu_id}_{stage}', 
                        f'Receive KV\nGPU {gpu_id}\nFrom GPU {(gpu_id-1)%num_gpus}\n(B, L/4, d_model)', 
                        shape='diamond', style='filled', fillcolor='lightgray')
                
                dot.node(f'send_{gpu_id}_{stage}', 
                        f'Send KV\nGPU {gpu_id}\nTo GPU {next_gpu}\n(B, L/4, d_model)', 
                        shape='diamond', style='filled', fillcolor='lightgray')
    
    # Output aggregation and final processing
    for gpu_id in range(num_gpus):
        with dot.subgraph(name=f'cluster_output_{gpu_id}') as c:
            c.attr(label=f'Output GPU {gpu_id}', style='rounded', bgcolor=colors[gpu_id])
            
            # Concatenate heads
            c.node(f'concat_{gpu_id}', f'Concatenate Heads\nGPU {gpu_id}\n(B, L/4, d_model)', 
                   shape='parallelogram', style='filled', fillcolor='white')
            
            # Output projection
            c.node(f'out_proj_{gpu_id}', f'Output Projection\nGPU {gpu_id}\n(B, L/4, d_model)', 
                   shape='rectangle', style='filled', fillcolor='white')
            
            # Residual connection
            c.node(f'residual_{gpu_id}', f'Residual Add\nGPU {gpu_id}\n(B, L/4, d_model)', 
                   shape='ellipse', style='filled', fillcolor='white')
            
            # Final output
            c.node(f'output_{gpu_id}', f'Output\nGPU {gpu_id}\n(B, L/4, d_model)', 
                   shape='ellipse', style='filled', fillcolor='white')
    
    # Connect the DAG
    
    # Input to projections
    for gpu_id in range(num_gpus):
        dot.edge(f'input_{gpu_id}', f'q_proj_{gpu_id}')
        dot.edge(f'input_{gpu_id}', f'k_proj_{gpu_id}')
        dot.edge(f'input_{gpu_id}', f'v_proj_{gpu_id}')
        
        # Projections to head splitting
        dot.edge(f'q_proj_{gpu_id}', f'q_split_{gpu_id}')
        dot.edge(f'k_proj_{gpu_id}', f'k_split_{gpu_id}')
        dot.edge(f'v_proj_{gpu_id}', f'v_split_{gpu_id}')
    
    # Ring attention connections
    for stage in range(num_gpus):
        for gpu_id in range(num_gpus):
            src_idx = (gpu_id - stage) % num_gpus
            
            # KV routing for attention
            if stage == 0:
                dot.edge(f'k_split_{src_idx}', f'attn_{gpu_id}_{stage}')
                dot.edge(f'v_split_{src_idx}', f'attn_{gpu_id}_{stage}')
            else:
                dot.edge(f'comm_{gpu_id}_{stage}', f'attn_{gpu_id}_{stage}')
            
            dot.edge(f'q_split_{gpu_id}', f'attn_{gpu_id}_{stage}')
            
            # Accumulation chain
            if stage == 0:
                dot.edge(f'attn_{gpu_id}_{stage}', f'accum_{gpu_id}_{stage}')
            else:
                dot.edge(f'accum_{gpu_id}_{stage-1}', f'accum_{gpu_id}_{stage}')
                dot.edge(f'attn_{gpu_id}_{stage}', f'accum_{gpu_id}_{stage}')
    
    # Communication ring connections
    for stage in range(num_gpus - 1):
        for gpu_id in range(num_gpus):
            next_gpu = (gpu_id + 1) % num_gpus
            prev_gpu = (gpu_id - 1) % num_gpus
            
            # KV blocks flowing in ring
            if stage == 0:
                dot.edge(f'k_split_{gpu_id}', f'send_{gpu_id}_{stage}')
                dot.edge(f'v_split_{gpu_id}', f'send_{gpu_id}_{stage}')
            else:
                dot.edge(f'comm_{gpu_id}_{stage-1}', f'send_{gpu_id}_{stage}')
            
            dot.edge(f'send_{prev_gpu}_{stage}', f'comm_{gpu_id}_{stage}')
    
    # Output processing
    for gpu_id in range(num_gpus):
        dot.edge(f'accum_{gpu_id}_{num_gpus-1}', f'concat_{gpu_id}')
        dot.edge(f'concat_{gpu_id}', f'out_proj_{gpu_id}')
        dot.edge(f'out_proj_{gpu_id}', f'residual_{gpu_id}')
        dot.edge(f'input_{gpu_id}', f'residual_{gpu_id}')  # Residual connection
        dot.edge(f'residual_{gpu_id}', f'output_{gpu_id}')
    
    return dot

def create_moe_ring_attention_dag():
    """Create DAG for MoE model with Ring Attention + Sequence Parallelism"""
    
    dot = graphviz.Digraph(comment='MoE Ring Attention + Sequence Parallelism DAG')
    dot.attr(rankdir='TB', size='25,25')
    
    colors = {
        0: 'lightblue',
        1: 'lightgreen', 
        2: 'lightyellow',
        3: 'lightcoral'
    }
    
    num_gpus = 4
    num_experts = 8
    
    # Input and attention (same as dense model)
    for gpu_id in range(num_gpus):
        with dot.subgraph(name=f'cluster_gpu_{gpu_id}') as c:
            c.attr(label=f'GPU {gpu_id}', style='rounded', bgcolor=colors[gpu_id])
            
            # Input
            c.node(f'input_{gpu_id}', f'Input\nX[{gpu_id}]\n(B, L/4, d_model)', 
                   shape='ellipse', style='filled', fillcolor='white')
            
            # Attention (reuse ring attention structure)
            c.node(f'attn_out_{gpu_id}', f'Attention Output\n(B, L/4, d_model)', 
                   shape='rectangle', style='filled', fillcolor='white')
            
            # Layer norm
            c.node(f'ln1_{gpu_id}', f'LayerNorm\n(B, L/4, d_model)', 
                   shape='rectangle', style='filled', fillcolor='white')
            
            # MoE components
            c.node(f'gate_{gpu_id}', f'Gate\n(B, L/4, num_experts)', 
                   shape='parallelogram', style='filled', fillcolor='lightpink')
            
            # Expert routing (dashed for selection)
            for expert_id in range(num_experts):
                if expert_id % num_gpus == gpu_id:
                    c.node(f'expert_{gpu_id}_{expert_id}', 
                           f'Expert {expert_id}\nMLP\n(B, k, d_model)', 
                           shape='rectangle', style='filled', fillcolor='lightcyan')
    
    # Expert selection and routing
    for gpu_id in range(num_gpus):
        # Gate computes routing decisions
        dot.edge(f'ln1_{gpu_id}', f'gate_{gpu_id}')
        
        # Route to appropriate experts (dashed lines for selection)
        for expert_id in range(num_experts):
            target_gpu = expert_id % num_gpus
            if gpu_id == target_gpu:
                # Local expert
                dot.edge(f'gate_{gpu_id}', f'expert_{gpu_id}_{expert_id}', 
                        style='dashed', label=f'select expert {expert_id}')
                dot.edge(f'ln1_{gpu_id}', f'expert_{gpu_id}_{expert_id}')
            else:
                # Remote expert - need communication
                dot.node(f'send_expert_{gpu_id}_{expert_id}', 
                        f'Send to Expert {expert_id}\n(B, k, d_model)', 
                        shape='diamond', style='filled', fillcolor='lightgray')
                dot.node(f'recv_expert_{target_gpu}_{expert_id}', 
                        f'Recv from GPU {gpu_id}\n(B, k, d_model)', 
                        shape='diamond', style='filled', fillcolor='lightgray')
                
                dot.edge(f'gate_{gpu_id}', f'send_expert_{gpu_id}_{expert_id}', 
                        style='dashed', label=f'select expert {expert_id}')
                dot.edge(f'ln1_{gpu_id}', f'send_expert_{gpu_id}_{expert_id}')
                dot.edge(f'send_expert_{gpu_id}_{expert_id}', f'recv_expert_{target_gpu}_{expert_id}')
                dot.edge(f'recv_expert_{target_gpu}_{expert_id}', f'expert_{target_gpu}_{expert_id}')
    
    # Expert outputs and aggregation
    for gpu_id in range(num_gpus):
        with dot.subgraph(name=f'cluster_output_{gpu_id}') as c:
            c.attr(label=f'Output GPU {gpu_id}', style='rounded', bgcolor=colors[gpu_id])
            
            # Aggregate expert outputs
            c.node(f'aggregate_{gpu_id}', f'Aggregate\nExpert Outputs\n(B, L/4, d_model)', 
                   shape='parallelogram', style='filled', fillcolor='white')
            
            # Final layer norm and residual
            c.node(f'ln2_{gpu_id}', f'LayerNorm\n(B, L/4, d_model)', 
                   shape='rectangle', style='filled', fillcolor='white')
            c.node(f'residual_{gpu_id}', f'Residual Add\n(B, L/4, d_model)', 
                   shape='ellipse', style='filled', fillcolor='white')
            c.node(f'output_{gpu_id}', f'Output\n(B, L/4, d_model)', 
                   shape='ellipse', style='filled', fillcolor='white')
    
    # Connect expert outputs
    for gpu_id in range(num_gpus):
        for expert_id in range(num_experts):
            if expert_id % num_gpus == gpu_id:
                dot.edge(f'expert_{gpu_id}_{expert_id}', f'aggregate_{gpu_id}')
        
        dot.edge(f'aggregate_{gpu_id}', f'ln2_{gpu_id}')
        dot.edge(f'ln2_{gpu_id}', f'residual_{gpu_id}')
        dot.edge(f'attn_out_{gpu_id}', f'residual_{gpu_id}')  # Residual from attention
        dot.edge(f'residual_{gpu_id}', f'output_{gpu_id}')
    
    return dot

# Generate both DAGs
if __name__ == "__main__":
    # Dense transformer DAG
    dense_dag = create_ring_attention_dag()
    dense_dag.render('/home/wzc/data/file-share/submission/ring_attention_dense', format='svg', cleanup=True)
    
    # MoE transformer DAG  
    moe_dag = create_moe_ring_attention_dag()
    moe_dag.render('/home/wzc/data/file-share/submission/ring_attention_moe', format='svg', cleanup=True)
    
    print("DAGs generated successfully!")
    print("Dense model DAG: /home/wzc/data/file-share/submission/ring_attention_dense.svg")
    print("MoE model DAG: /home/wzc/data/file-share/submission/ring_attention_moe.svg")