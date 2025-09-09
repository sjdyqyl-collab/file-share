#!/usr/bin/env python3

import graphviz

def create_baseline_dag():
    """
    Create baseline DAG for TP=8, PP=2 with 16 GPUs
    Each GPU has 4 experts + 1/8 tensor parallel shard
    """
    
    dot = graphviz.Digraph('baseline_moe_dag', comment='Baseline MoE DAG - TP=8, PP=2, 16 GPUs')
    dot.attr(rankdir='TB', size='20,30')
    
    # Define global dimensions
    batch_size = 1024
    seq_len = 10000
    hidden_size = 8192
    mlp_hidden_size = 32768
    num_heads = 16
    head_dim = 512
    num_experts = 16
    experts_per_gpu = 4
    tp_size = 8
    pp_stages = 2
    
    # Input node
    dot.node('input', f'Total Input\\nbatch={batch_size}, seq={seq_len}, hidden={hidden_size}', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Pipeline Stage 0 (GPUs 0-7)
    with dot.subgraph(name='cluster_pipeline_0') as p0:
        p0.attr(label='Pipeline Stage 0 (GPUs 0-7)', style='rounded', fillcolor='lightgray')
        
        # Layer 0
        for gpu_id in range(8):
            gpu_name = f'gpu_{gpu_id}'
            with p0.subgraph(name=f'cluster_{gpu_name}_layer0') as gpu_cluster:
                gpu_cluster.attr(label=f'GPU {gpu_name} - Layer 0', style='rounded')
                
                # MHA - Tensor Parallel
                gpu_cluster.node(f'mha_qkv_0_{gpu_id}', 
                               f'MHA QKV Linear\\nINPUT: {batch_size},{seq_len},{hidden_size}\\nOUTPUT: {batch_size},{seq_len},{hidden_size//tp_size*3}\\nGPU: {gpu_id}',
                               shape='rectangle', style='filled', fillcolor='lightgreen')
                
                gpu_cluster.node(f'mha_split_0_{gpu_id}',
                               f'Split Heads\\nINPUT: {batch_size},{seq_len},{hidden_size//tp_size*3}\\nOUTPUT: {batch_size},{num_heads//tp_size},{seq_len},{head_dim*3}\\nGPU: {gpu_id}',
                               shape='parallelogram', style='filled', fillcolor='yellow')
                
                gpu_cluster.node(f'mha_attn_0_{gpu_id}',
                               f'MHA Attention\\nINPUT: {batch_size},{num_heads//tp_size},{seq_len},{head_dim}\\nOUTPUT: {batch_size},{num_heads//tp_size},{seq_len},{head_dim}\\nGPU: {gpu_id}',
                               shape='rectangle', style='filled', fillcolor='lightgreen')
                
                gpu_cluster.node(f'mha_concat_0_{gpu_id}',
                               f'Concat Heads\\nINPUT: {batch_size},{num_heads//tp_size},{seq_len},{head_dim}\\nOUTPUT: {batch_size},{seq_len},{hidden_size//tp_size}\\nGPU: {gpu_id}',
                               shape='parallelogram', style='filled', fillcolor='yellow')
                
                gpu_cluster.node(f'mha_out_0_{gpu_id}',
                               f'MHA Output Linear\\nINPUT: {batch_size},{seq_len},{hidden_size//tp_size}\\nOUTPUT: {batch_size},{seq_len},{hidden_size//tp_size}\\nGPU: {gpu_id}',
                               shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # MHA All-reduce
                gpu_cluster.node(f'mha_allreduce_0_{gpu_id}',
                               f'MHA All-Reduce\\nINPUT: {batch_size},{seq_len},{hidden_size//tp_size}\\nOUTPUT: {batch_size},{seq_len},{hidden_size}\\nGPU: {gpu_id}',
                               shape='ellipse', style='filled', fillcolor='orange')
                
                # Residual Add 1
                gpu_cluster.node(f'residual1_0_{gpu_id}',
                               f'Residual Add 1\\nINPUT: {batch_size},{seq_len},{hidden_size}\\nOUTPUT: {batch_size},{seq_len},{hidden_size}\\nGPU: {gpu_id}',
                               shape='diamond', style='filled', fillcolor='pink')
                
                # MoE Layer - 4 experts per GPU
                for expert_idx in range(experts_per_gpu):
                    expert_id = gpu_id * experts_per_gpu + expert_idx
                    gpu_cluster.node(f'expert_gate_0_{gpu_id}_{expert_idx}',
                                   f'Expert {expert_id} Gate\\nINPUT: {batch_size},{seq_len},{hidden_size}\\nOUTPUT: routing decisions\\nGPU: {gpu_id}',
                                   shape='parallelogram', style='filled', fillcolor='lightcyan')
                    
                    gpu_cluster.node(f'expert_linear1_0_{gpu_id}_{expert_idx}',
                                   f'Expert {expert_id} Linear 1\\nINPUT: {batch_size},{seq_len},{hidden_size}\\nOUTPUT: {batch_size},{seq_len},{mlp_hidden_size}\\nGPU: {gpu_id}',
                                   shape='rectangle', style='filled', fillcolor='lightgreen')
                    
                    gpu_cluster.node(f'expert_activation_0_{gpu_id}_{expert_idx}',
                                   f'Expert {expert_id} GELU\\nINPUT: {batch_size},{seq_len},{mlp_hidden_size}\\nOUTPUT: {batch_size},{seq_len},{mlp_hidden_size}\\nGPU: {gpu_id}',
                                   shape='rectangle', style='filled', fillcolor='lightgreen')
                    
                    gpu_cluster.node(f'expert_linear2_0_{gpu_id}_{expert_idx}',
                                   f'Expert {expert_id} Linear 2\\nINPUT: {batch_size},{seq_len},{mlp_hidden_size}\\nOUTPUT: {batch_size},{seq_len},{hidden_size}\\nGPU: {gpu_id}',
                                   shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # Expert aggregation
                gpu_cluster.node(f'expert_aggregate_0_{gpu_id}',
                               f'Expert Aggregate\\nINPUT: {batch_size},{seq_len},{hidden_size}\\nOUTPUT: {batch_size},{seq_len},{hidden_size}\\nGPU: {gpu_id}',
                               shape='parallelogram', style='filled', fillcolor='yellow')
                
                # Residual Add 2
                gpu_cluster.node(f'residual2_0_{gpu_id}',
                               f'Residual Add 2\\nINPUT: {batch_size},{seq_len},{hidden_size}\\nOUTPUT: {batch_size},{seq_len},{hidden_size}\\nGPU: {gpu_id}',
                               shape='diamond', style='filled', fillcolor='pink')
    
    # Pipeline communication between stages
    dot.node('pipeline_comm_0_1', 'Pipeline Communication\\nStage 0 -> Stage 1\\nbatch=1024, seq=10000, hidden=8192\\nCross-GPU Transfer',
             shape='ellipse', style='filled', fillcolor='red')
    
    # Pipeline Stage 1 (GPUs 8-15)
    with dot.subgraph(name='cluster_pipeline_1') as p1:
        p1.attr(label='Pipeline Stage 1 (GPUs 8-15)', style='rounded', fillcolor='lightgray')
        
        # Layer 1
        for gpu_id in range(8, 16):
            actual_gpu_id = gpu_id - 8
            gpu_name = f'gpu_{gpu_id}'
            with p1.subgraph(name=f'cluster_{gpu_name}_layer1') as gpu_cluster:
                gpu_cluster.attr(label=f'GPU {gpu_name} - Layer 1', style='rounded')
                
                # MHA - Tensor Parallel
                gpu_cluster.node(f'mha_qkv_1_{gpu_id}', 
                               f'MHA QKV Linear\\nINPUT: {batch_size},{seq_len},{hidden_size}\\nOUTPUT: {batch_size},{seq_len},{hidden_size//tp_size*3}\\nGPU: {gpu_id}',
                               shape='rectangle', style='filled', fillcolor='lightgreen')
                
                gpu_cluster.node(f'mha_split_1_{gpu_id}',
                               f'Split Heads\\nINPUT: {batch_size},{seq_len},{hidden_size//tp_size*3}\\nOUTPUT: {batch_size},{num_heads//tp_size},{seq_len},{head_dim*3}\\nGPU: {gpu_id}',
                               shape='parallelogram', style='filled', fillcolor='yellow')
                
                gpu_cluster.node(f'mha_attn_1_{gpu_id}',
                               f'MHA Attention\\nINPUT: {batch_size},{num_heads//tp_size},{seq_len},{head_dim}\\nOUTPUT: {batch_size},{num_heads//tp_size},{seq_len},{head_dim}\\nGPU: {gpu_id}',
                               shape='rectangle', style='filled', fillcolor='lightgreen')
                
                gpu_cluster.node(f'mha_concat_1_{gpu_id}',
                               f'Concat Heads\\nINPUT: {batch_size},{num_heads//tp_size},{seq_len},{head_dim}\\nOUTPUT: {batch_size},{seq_len},{hidden_size//tp_size}\\nGPU: {gpu_id}',
                               shape='parallelogram', style='filled', fillcolor='yellow')
                
                gpu_cluster.node(f'mha_out_1_{gpu_id}',
                               f'MHA Output Linear\\nINPUT: {batch_size},{seq_len},{hidden_size//tp_size}\\nOUTPUT: {batch_size},{seq_len},{hidden_size//tp_size}\\nGPU: {gpu_id}',
                               shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # MHA All-reduce
                gpu_cluster.node(f'mha_allreduce_1_{gpu_id}',
                               f'MHA All-Reduce\\nINPUT: {batch_size},{seq_len},{hidden_size//tp_size}\\nOUTPUT: {batch_size},{seq_len},{hidden_size}\\nGPU: {gpu_id}',
                               shape='ellipse', style='filled', fillcolor='orange')
                
                # Residual Add 1
                gpu_cluster.node(f'residual1_1_{gpu_id}',
                               f'Residual Add 1\\nINPUT: {batch_size},{seq_len},{hidden_size}\\nOUTPUT: {batch_size},{seq_len},{hidden_size}\\nGPU: {gpu_id}',
                               shape='diamond', style='filled', fillcolor='pink')
                
                # MoE Layer - 4 experts per GPU
                for expert_idx in range(experts_per_gpu):
                    expert_id = (gpu_id - 8) * experts_per_gpu + expert_idx
                    gpu_cluster.node(f'expert_gate_1_{gpu_id}_{expert_idx}',
                                   f'Expert {expert_id} Gate\\nINPUT: {batch_size},{seq_len},{hidden_size}\\nOUTPUT: routing decisions\\nGPU: {gpu_id}',
                                   shape='parallelogram', style='filled', fillcolor='lightcyan')
                    
                    gpu_cluster.node(f'expert_linear1_1_{gpu_id}_{expert_idx}',
                                   f'Expert {expert_id} Linear 1\\nINPUT: {batch_size},{seq_len},{hidden_size}\\nOUTPUT: {batch_size},{seq_len},{mlp_hidden_size}\\nGPU: {gpu_id}',
                                   shape='rectangle', style='filled', fillcolor='lightgreen')
                    
                    gpu_cluster.node(f'expert_activation_1_{gpu_id}_{expert_idx}',
                                   f'Expert {expert_id} GELU\\nINPUT: {batch_size},{seq_len},{mlp_hidden_size}\\nOUTPUT: {batch_size},{seq_len},{mlp_hidden_size}\\nGPU: {gpu_id}',
                                   shape='rectangle', style='filled', fillcolor='lightgreen')
                    
                    gpu_cluster.node(f'expert_linear2_1_{gpu_id}_{expert_idx}',
                                   f'Expert {expert_id} Linear 2\\nINPUT: {batch_size},{seq_len},{mlp_hidden_size}\\nOUTPUT: {batch_size},{seq_len},{hidden_size}\\nGPU: {gpu_id}',
                                   shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # Expert aggregation
                gpu_cluster.node(f'expert_aggregate_1_{gpu_id}',
                               f'Expert Aggregate\\nINPUT: {batch_size},{seq_len},{hidden_size}\\nOUTPUT: {batch_size},{seq_len},{hidden_size}\\nGPU: {gpu_id}',
                               shape='parallelogram', style='filled', fillcolor='yellow')
                
                # Residual Add 2
                gpu_cluster.node(f'residual2_1_{gpu_id}',
                               f'Residual Add 2\\nINPUT: {batch_size},{seq_len},{hidden_size}\\nOUTPUT: {batch_size},{seq_len},{hidden_size}\\nGPU: {gpu_id}',
                               shape='diamond', style='filled', fillcolor='pink')
    
    # Output node
    dot.node('output', f'Total Output\\nbatch={batch_size}, seq={seq_len}, hidden={hidden_size}', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connect input to first layer
    for gpu_id in range(8):
        dot.edge('input', f'mha_qkv_0_{gpu_id}')
    
    # Connect within layer 0
    for gpu_id in range(8):
        dot.edge(f'mha_qkv_0_{gpu_id}', f'mha_split_0_{gpu_id}')
        dot.edge(f'mha_split_0_{gpu_id}', f'mha_attn_0_{gpu_id}')
        dot.edge(f'mha_attn_0_{gpu_id}', f'mha_concat_0_{gpu_id}')
        dot.edge(f'mha_concat_0_{gpu_id}', f'mha_out_0_{gpu_id}')
        dot.edge(f'mha_out_0_{gpu_id}', f'mha_allreduce_0_{gpu_id}')
        dot.edge(f'mha_allreduce_0_{gpu_id}', f'residual1_0_{gpu_id}')
        
        # Connect to experts
        for expert_idx in range(experts_per_gpu):
            dot.edge(f'residual1_0_{gpu_id}', f'expert_gate_0_{gpu_id}_{expert_idx}', style='dashed')
            dot.edge(f'expert_gate_0_{gpu_id}_{expert_idx}', f'expert_linear1_0_{gpu_id}_{expert_idx}')
            dot.edge(f'expert_linear1_0_{gpu_id}_{expert_idx}', f'expert_activation_0_{gpu_id}_{expert_idx}')
            dot.edge(f'expert_activation_0_{gpu_id}_{expert_idx}', f'expert_linear2_0_{gpu_id}_{expert_idx}')
            dot.edge(f'expert_linear2_0_{gpu_id}_{expert_idx}', f'expert_aggregate_0_{gpu_id}')
        
        dot.edge(f'expert_aggregate_0_{gpu_id}', f'residual2_0_{gpu_id}')
        dot.edge(f'residual2_0_{gpu_id}', 'pipeline_comm_0_1')
    
    # Connect pipeline stages
    for gpu_id in range(8, 16):
        dot.edge('pipeline_comm_0_1', f'mha_qkv_1_{gpu_id}')
    
    # Connect within layer 1
    for gpu_id in range(8, 16):
        dot.edge(f'mha_qkv_1_{gpu_id}', f'mha_split_1_{gpu_id}')
        dot.edge(f'mha_split_1_{gpu_id}', f'mha_attn_1_{gpu_id}')
        dot.edge(f'mha_attn_1_{gpu_id}', f'mha_concat_1_{gpu_id}')
        dot.edge(f'mha_concat_1_{gpu_id}', f'mha_out_1_{gpu_id}')
        dot.edge(f'mha_out_1_{gpu_id}', f'mha_allreduce_1_{gpu_id}')
        dot.edge(f'mha_allreduce_1_{gpu_id}', f'residual1_1_{gpu_id}')
        
        # Connect to experts
        for expert_idx in range(experts_per_gpu):
            dot.edge(f'residual1_1_{gpu_id}', f'expert_gate_1_{gpu_id}_{expert_idx}', style='dashed')
            dot.edge(f'expert_gate_1_{gpu_id}_{expert_idx}', f'expert_linear1_1_{gpu_id}_{expert_idx}')
            dot.edge(f'expert_linear1_1_{gpu_id}_{expert_idx}', f'expert_activation_1_{gpu_id}_{expert_idx}')
            dot.edge(f'expert_activation_1_{gpu_id}_{expert_idx}', f'expert_linear2_1_{gpu_id}_{expert_idx}')
            dot.edge(f'expert_linear2_1_{gpu_id}_{expert_idx}', f'expert_aggregate_1_{gpu_id}')
        
        dot.edge(f'expert_aggregate_1_{gpu_id}', f'residual2_1_{gpu_id}')
        dot.edge(f'residual2_1_{gpu_id}', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/2025-09-09-11-32-31/baseline_moe_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-09-11-32-31/baseline_moe_dag.dot')
    print("Baseline DAG generated successfully")