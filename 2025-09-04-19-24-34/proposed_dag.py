#!/usr/bin/env python3

import graphviz

def create_proposed_dag():
    dot = graphviz.Digraph('Proposed_Cross_Node_Expert_Parallelism')
    dot.attr(rankdir='TB', size='30,40')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Create clusters for each GPU (0-63)
    for gpu_id in range(64):
        with dot.subgraph(name=f'cluster_gpu_{gpu_id}') as cluster:
            cluster.attr(label=f'GPU {gpu_id}', style='dashed', color='green')
            
            # For each GPU, we have 1 expert from each of the 4 layers
            for layer_id in range(4):
                expert_id = gpu_id  # Each GPU gets one unique expert per layer
                exp_node = f'layer_{layer_id}_expert_{expert_id}_gpu_{gpu_id}'
                cluster.node(exp_node, 
                           f'Layer {layer_id} Expert {expert_id}\nGPU {gpu_id}\nMLP Expert\nInput: [tokens, 8192]\nOutput: [tokens, 8192]', 
                           shape='rectangle', fillcolor='pink')
    
    # Create MHA and routing nodes
    for layer_id in range(4):
        with dot.subgraph(name=f'cluster_layer_{layer_id}_mha') as mha_cluster:
            mha_cluster.attr(label=f'Layer {layer_id} MHA', style='dotted', color='blue')
            
            # MHA nodes (replicated across all GPUs for load balancing)
            for gpu_id in range(64):
                mha_node = f'mha_{layer_id}_gpu_{gpu_id}'
                mha_cluster.node(mha_node, 
                               f'MHA Layer {layer_id}\nGPU {gpu_id}\nInput: [tokens, 8192]\nOutput: [tokens, 8192]', 
                               shape='ellipse', fillcolor='lightgreen')
                
                # Residual connections
                res_node = f'res_{layer_id}_gpu_{gpu_id}'
                mha_cluster.node(res_node, 
                               f'Residual Add\nGPU {gpu_id}\nInput: [tokens, 8192] × 2\nOutput: [tokens, 8192]', 
                               shape='parallelogram', fillcolor='orange')
                
                # Gate for expert selection
                gate_node = f'gate_{layer_id}_gpu_{gpu_id}'
                mha_cluster.node(gate_node, 
                               f'Expert Gate\nLayer {layer_id}\nGPU {gpu_id}\nInput: [tokens, 8192]\nOutput: [tokens, 16] (routing)', 
                               shape='diamond', fillcolor='purple', style='dashed')
                
                # Token routing/splitting
                split_node = f'split_{layer_id}_gpu_{gpu_id}'
                mha_cluster.node(split_node, 
                               f'Token Split\nLayer {layer_id}\nGPU {gpu_id}\nInput: [tokens, 8192]\nOutput: [tokens/16, 8192] × 16', 
                               shape='parallelogram', fillcolor='yellow')
                
                # Expert aggregation
                agg_node = f'agg_{layer_id}_gpu_{gpu_id}'
                mha_cluster.node(agg_node, 
                               f'Expert Aggregation\nLayer {layer_id}\nGPU {gpu_id}\nInput: [tokens/16, 8192] × 16\nOutput: [tokens, 8192]', 
                               shape='parallelogram', fillcolor='orange')
                
                # Second residual
                res2_node = f'res2_{layer_id}_gpu_{gpu_id}'
                mha_cluster.node(res2_node, 
                               f'Residual Add\nLayer {layer_id}\nGPU {gpu_id}\nInput: [tokens, 8192] × 2\nOutput: [tokens, 8192]', 
                               shape='parallelogram', fillcolor='orange')
    
    # Communication nodes for cross-GPU token routing
    for layer_id in range(4):
        for src_gpu in range(64):
            for dst_gpu in range(64):
                if src_gpu != dst_gpu:
                    comm_node = f'comm_{layer_id}_{src_gpu}_to_{dst_gpu}'
                    dot.node(comm_node, 
                           f'Token Transfer\nLayer {layer_id}\nGPU {src_gpu} → GPU {dst_gpu}\nInput: [tokens/16, 8192]\nOutput: [tokens/16, 8192]', 
                           shape='ellipse', fillcolor='red', style='dashed')
    
    # Input and output nodes
    dot.node('input', 'Model Input\n[1024, 8192]\nDistributed to all GPUs', shape='ellipse', fillcolor='white')
    dot.node('output', 'Model Output\n[1024, 8192]\nFrom all GPUs', shape='ellipse', fillcolor='white')
    
    # Create edges for the complete flow
    # Input to first layer MHA
    for gpu_id in range(64):
        dot.edge('input', f'mha_0_gpu_{gpu_id}')
        dot.edge(f'mha_0_gpu_{gpu_id}', f'res_0_gpu_{gpu_id}')
        dot.edge(f'res_0_gpu_{gpu_id}', f'gate_0_gpu_{gpu_id}')
        dot.edge(f'gate_0_gpu_{gpu_id}', f'split_0_gpu_{gpu_id}')
        
        # Connect split to experts across all GPUs
        for expert_id in range(64):
            src_gpu = gpu_id
            dst_gpu = expert_id
            if src_gpu == dst_gpu:
                # Local expert processing
                dot.edge(f'split_0_gpu_{src_gpu}', f'layer_0_expert_{expert_id}_gpu_{dst_gpu}')
            else:
                # Cross-GPU communication
                comm_node = f'comm_0_{src_gpu}_to_{dst_gpu}'
                dot.edge(f'split_0_gpu_{src_gpu}', comm_node)
                dot.edge(comm_node, f'layer_0_expert_{expert_id}_gpu_{dst_gpu}')
        
        # Connect experts back to aggregation
        for expert_id in range(64):
            dst_gpu = gpu_id
            src_gpu = expert_id
            if src_gpu == dst_gpu:
                dot.edge(f'layer_0_expert_{expert_id}_gpu_{src_gpu}', f'agg_0_gpu_{dst_gpu}')
            else:
                comm_node = f'comm_0_{src_gpu}_to_{dst_gpu}'
                dot.edge(f'layer_0_expert_{expert_id}_gpu_{src_gpu}', comm_node)
                dot.edge(comm_node, f'agg_0_gpu_{dst_gpu}')
        
        dot.edge(f'agg_0_gpu_{gpu_id}', f'res2_0_gpu_{gpu_id}')
        
        # Connect to next layer MHA
        dot.edge(f'res2_0_gpu_{gpu_id}', f'mha_1_gpu_{gpu_id}')
        
        # Repeat for layers 1, 2, 3
        for layer_id in range(1, 4):
            dot.edge(f'mha_{layer_id}_gpu_{gpu_id}', f'res_{layer_id}_gpu_{gpu_id}')
            dot.edge(f'res_{layer_id}_gpu_{gpu_id}', f'gate_{layer_id}_gpu_{gpu_id}')
            dot.edge(f'gate_{layer_id}_gpu_{gpu_id}', f'split_{layer_id}_gpu_{gpu_id}')
            
            for expert_id in range(64):
                src_gpu = gpu_id
                dst_gpu = expert_id
                if src_gpu == dst_gpu:
                    dot.edge(f'split_{layer_id}_gpu_{src_gpu}', f'layer_{layer_id}_expert_{expert_id}_gpu_{dst_gpu}')
                else:
                    comm_node = f'comm_{layer_id}_{src_gpu}_to_{dst_gpu}'
                    dot.edge(f'split_{layer_id}_gpu_{src_gpu}', comm_node)
                    dot.edge(comm_node, f'layer_{layer_id}_expert_{expert_id}_gpu_{dst_gpu}')
            
            for expert_id in range(64):
                dst_gpu = gpu_id
                src_gpu = expert_id
                if src_gpu == dst_gpu:
                    dot.edge(f'layer_{layer_id}_expert_{expert_id}_gpu_{src_gpu}', f'agg_{layer_id}_gpu_{dst_gpu}')
                else:
                    comm_node = f'comm_{layer_id}_{src_gpu}_to_{dst_gpu}'
                    dot.edge(f'layer_{layer_id}_expert_{expert_id}_gpu_{src_gpu}', comm_node)
                    dot.edge(comm_node, f'agg_{layer_id}_gpu_{dst_gpu}')
            
            dot.edge(f'agg_{layer_id}_gpu_{gpu_id}', f'res2_{layer_id}_gpu_{gpu_id}')
            
            if layer_id < 3:
                dot.edge(f'res2_{layer_id}_gpu_{gpu_id}', f'mha_{layer_id+1}_gpu_{gpu_id}')
            else:
                dot.edge(f'res2_3_gpu_{gpu_id}', 'output')
    
    # Save the DAG
    dot.format = 'svg'
    dot.render('/home/wzc/data/file-share/2025-09-04-19-24-34/proposed_moe_dag', cleanup=False)
    
    # Also save as .dot file
    with open('/home/wzc/data/file-share/2025-09-04-19-24-34/proposed_moe_dag.dot', 'w') as f:
        f.write(dot.source)

if __name__ == '__main__':
    create_proposed_dag()