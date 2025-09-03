import graphviz
from typing import List, Dict, Tuple
import json

def create_layer_wise_dag():
    """Create DAG for proposed layer-wise deployment strategy"""
    dot = graphviz.Digraph('LayerWiseDeployment', format='svg')
    dot.attr(rankdir='TB', size='20,30')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input\n[1024, 8192]\nAll GPUs', shape='ellipse', fillcolor='lightgreen')
    
    # Create 16 layers, each on separate GPU
    for i in range(16):
        gpu_id = i
        
        # Layer computation node
        layer_name = f'layer_{i}'
        dot.node(layer_name, 
                f'Layer {i}\n[1024, 8192] → [1024, 8192]\nGPU {gpu_id}', 
                fillcolor='lightblue')
        
        # Communication nodes for data transfer
        if i > 0:
            transfer_name = f'transfer_{i-1}_{i}'
            dot.node(transfer_name, 
                    f'Transfer\n[1024, 8192]\nGPU {i-1} → GPU {i}', 
                    shape='parallelogram', fillcolor='yellow')
            dot.edge(f'layer_{i-1}', transfer_name)
            dot.edge(transfer_name, layer_name)
        else:
            # First layer connects to input
            dot.edge('input', layer_name)
    
    # Output node
    dot.node('output', 'Output\n[1024, 8192]\nGPU 15', shape='ellipse', fillcolor='lightgreen')
    dot.edge('layer_15', 'output')
    
    return dot

def create_baseline_dag():
    """Create DAG for baseline tensor + pipeline parallelism"""
    dot = graphviz.Digraph('BaselineTensorPipeline', format='svg')
    dot.attr(rankdir='TB', size='30,40')
    
    # Color coding for tensor parallel groups
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue']
    
    # Input node
    dot.node('input', 'Input\n[1024, 8192]\nAll GPUs', shape='ellipse', fillcolor='lightgreen')
    
    # Pipeline Stage 0: Layers 0-7 on GPUs 0-7 (TP=8)
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0\nGPUs 0-7 (TP=8)', style='dashed', color='blue')
        
        for layer in range(8):
            # Tensor parallel split across 8 GPUs
            for gpu in range(8):
                node_id = f's0_l{layer}_gpu{gpu}'
                if layer == 0:
                    # First layer splits input
                    stage0.node(node_id, 
                              f'Layer {layer}\nTP Rank {gpu}\n[1024, 1024] → [1024, 1024]\nGPU {gpu}', 
                              fillcolor=colors[gpu])
                else:
                    stage0.node(node_id, 
                              f'Layer {layer}\nTP Rank {gpu}\n[1024, 1024] → [1024, 1024]\nGPU {gpu}', 
                              fillcolor=colors[gpu])
    
    # Pipeline Stage 1: Layers 8-15 on GPUs 8-15 (TP=8)
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1\nGPUs 8-15 (TP=8)', style='dashed', color='red')
        
        for layer in range(8, 16):
            # Tensor parallel split across 8 GPUs
            for gpu in range(8, 16):
                node_id = f's1_l{layer}_gpu{gpu}'
                stage1.node(node_id, 
                          f'Layer {layer}\nTP Rank {gpu-8}\n[1024, 1024] → [1024, 1024]\nGPU {gpu}', 
                          fillcolor=colors[gpu-8])
    
    # Add connections for tensor parallelism within each layer
    for layer in range(16):
        if layer < 8:
            # Stage 0 layers
            tp_devices = range(8)
            stage_prefix = 's0'
        else:
            # Stage 1 layers
            tp_devices = range(8, 16)
            stage_prefix = 's1'
            layer_idx = layer - 8
        
        # All-reduce communication for tensor parallelism
        if layer > 0 or layer >= 8:
            # Add tensor parallel communication nodes
            allreduce_node = f'allreduce_l{layer}'
            if layer < 8:
                dot.node(allreduce_node, 
                        f'All-Reduce\n[1024, 8192]\nGPUs 0-7', 
                        shape='parallelogram', fillcolor='orange')
            else:
                dot.node(allreduce_node, 
                        f'All-Reduce\n[1024, 8192]\nGPUs 8-15', 
                        shape='parallelogram', fillcolor='orange')
    
    # Connect layers within pipeline stages
    for layer in range(16):
        if layer < 8:
            # Stage 0
            if layer == 0:
                # Input to first layer (tensor parallel split)
                split_node = 'split_input_s0'
                dot.node(split_node, 'Split\n[1024, 8192] → 8×[1024, 1024]\nGPUs 0-7', 
                        shape='parallelogram', fillcolor='yellow')
                dot.edge('input', split_node)
                for gpu in range(8):
                    dot.edge(split_node, f's0_l0_gpu{gpu}')
            else:
                # Connect previous layer to current
                for gpu in range(8):
                    dot.edge(f's0_l{layer-1}_gpu{gpu}', f's0_l{layer}_gpu{gpu}')
        else:
            # Stage 1
            layer_idx = layer - 8
            if layer_idx == 0:
                # Pipeline transfer from stage 0 to stage 1
                transfer_node = 'transfer_s0_s1'
                dot.node(transfer_node, 
                        'Pipeline Transfer\n[1024, 8192]\nGPU 7 → GPU 8', 
                        shape='parallelogram', fillcolor='red')
                
                # Connect last layer of stage 0 to transfer
                for gpu in range(8):
                    dot.edge(f's0_l7_gpu{gpu}', transfer_node)
                
                # Split for stage 1
                split_node = 'split_input_s1'
                dot.node(split_node, 'Split\n[1024, 8192] → 8×[1024, 1024]\nGPUs 8-15', 
                        shape='parallelogram', fillcolor='yellow')
                dot.edge(transfer_node, split_node)
                
                for gpu in range(8, 16):
                    dot.edge(split_node, f's1_l8_gpu{gpu}')
            else:
                # Connect within stage 1
                for gpu in range(8, 16):
                    dot.edge(f's1_l{layer-1}_gpu{gpu}', f's1_l{layer}_gpu{gpu}')
    
    # Output aggregation
    output_node = 'output'
    dot.node(output_node, 'Output\n[1024, 8192]\nGPU 15', shape='ellipse', fillcolor='lightgreen')
    
    # Gather from final layer
    gather_node = 'gather_output'
    dot.node(gather_node, 'Gather\n8×[1024, 1024] → [1024, 8192]\nGPUs 8-15', 
            shape='parallelogram', fillcolor='yellow')
    
    for gpu in range(8, 16):
        dot.edge(f's1_l15_gpu{gpu}', gather_node)
    dot.edge(gather_node, output_node)
    
    return dot

def create_detailed_layer_dag(layer_id: int, gpu_id: int, strategy: str):
    """Create detailed DAG for a single layer showing internal operators"""
    dot = graphviz.Digraph(f'Layer{layer_id}_{strategy}', format='svg')
    dot.attr(rankdir='LR', size='15,10')
    
    # Input dimensions based on model specs
    batch_size = 1024
    seq_len = 1  # Assuming single token for simplicity
    hidden_size = 8192  # 16 heads * 512 dim per head
    mlp_hidden_size = 32768
    
    # Layer input
    dot.node('input', f'Layer {layer} Input\n[{batch_size}, {hidden_size}]', 
            shape='ellipse', fillcolor='lightgreen')
    
    # Multi-Head Attention
    dot.node('layernorm1', f'LayerNorm\n[{batch_size}, {hidden_size}] → [{batch_size}, {hidden_size}]\nGPU {gpu_id}', 
            fillcolor='lightblue')
    
    # Q, K, V projections (column parallel for baseline)
    dot.node('q_proj', f'Q Projection\n[{batch_size}, {hidden_size}] → [{batch_size}, {hidden_size}]\nGPU {gpu_id}', 
            fillcolor='lightcoral')
    dot.node('k_proj', f'K Projection\n[{batch_size}, {hidden_size}] → [{batch_size}, {hidden_size}]\nGPU {gpu_id}', 
            fillcolor='lightcoral')
    dot.node('v_proj', f'V Projection\n[{batch_size}, {hidden_size}] → [{batch_size}, {hidden_size}]\nGPU {gpu_id}', 
            fillcolor='lightcoral')
    
    # Attention computation
    dot.node('attention', f'Multi-Head Attention\n[{batch_size}, {hidden_size}] → [{batch_size}, {hidden_size}]\nGPU {gpu_id}', 
            fillcolor='lightyellow')
    
    # Output projection
    dot.node('out_proj', f'Output Projection\n[{batch_size}, {hidden_size}] → [{batch_size}, {hidden_size}]\nGPU {gpu_id}', 
            fillcolor='lightcoral')
    
    # Residual connection
    dot.node('residual1', f'Residual Add\n[{batch_size}, {hidden_size}] + [{batch_size}, {hidden_size}]\nGPU {gpu_id}', 
            fillcolor='lightgray')
    
    # MLP
    dot.node('layernorm2', f'LayerNorm\n[{batch_size}, {hidden_size}] → [{batch_size}, {hidden_size}]\nGPU {gpu_id}', 
            fillcolor='lightblue')
    
    # MLP layers
    dot.node('mlp_up', f'MLP Up\n[{batch_size}, {hidden_size}] → [{batch_size}, {mlp_hidden_size}]\nGPU {gpu_id}', 
            fillcolor='lightpink')
    dot.node('activation', f'GELU Activation\n[{batch_size}, {mlp_hidden_size}] → [{batch_size}, {mlp_hidden_size}]\nGPU {gpu_id}', 
            fillcolor='lightgreen')
    dot.node('mlp_down', f'MLP Down\n[{batch_size}, {mlp_hidden_size}] → [{batch_size}, {hidden_size}]\nGPU {gpu_id}', 
            fillcolor='lightpink')
    
    # Final residual
    dot.node('residual2', f'Residual Add\n[{batch_size}, {hidden_size}] + [{batch_size}, {hidden_size}]\nGPU {gpu_id}', 
            fillcolor='lightgray')
    
    dot.node('output', f'Layer {layer} Output\n[{batch_size}, {hidden_size}]', 
            shape='ellipse', fillcolor='lightgreen')
    
    # Connect the flow
    dot.edge('input', 'layernorm1')
    dot.edge('layernorm1', 'q_proj')
    dot.edge('layernorm1', 'k_proj')
    dot.edge('layernorm1', 'v_proj')
    dot.edge('q_proj', 'attention')
    dot.edge('k_proj', 'attention')
    dot.edge('v_proj', 'attention')
    dot.edge('attention', 'out_proj')
    dot.edge('out_proj', 'residual1')
    dot.edge('input', 'residual1')  # Residual connection
    dot.edge('residual1', 'layernorm2')
    dot.edge('layernorm2', 'mlp_up')
    dot.edge('mlp_up', 'activation')
    dot.edge('activation', 'mlp_down')
    dot.edge('mlp_down', 'residual2')
    dot.edge('residual1', 'residual2')  # Residual connection
    dot.edge('residual2', 'output')
    
    return dot

if __name__ == '__main__':
    # Generate all DAGs
    
    # 1. Proposed layer-wise deployment
    layer_wise_dag = create_layer_wise_dag()
    layer_wise_dag.render('/home/wzc/data/file-share/2025-09-03-14-08-07/layer_wise_deployment')
    
    # 2. Baseline tensor + pipeline parallelism
    baseline_dag = create_baseline_dag()
    baseline_dag.render('/home/wzc/data/file-share/2025-09-03-14-08-07/baseline_tensor_pipeline')
    
    # 3. Detailed layer DAGs for both strategies
    for strategy in ['layer_wise', 'baseline']:
        for layer in [0, 7, 15]:  # First, middle, and last layers
            if strategy == 'layer_wise':
                gpu_id = layer
            else:
                gpu_id = 0 if layer < 8 else 8  # Representative GPU for tensor parallel group
            
            detailed_dag = create_detailed_layer_dag(layer, gpu_id, strategy)
            detailed_dag.render(f'/home/wzc/data/file-share/2025-09-03-14-08-07/detailed_layer_{layer}_{strategy}')
    
    # Save DOT files
    with open('/home/wzc/data/file-share/2025-09-03-14-08-07/layer_wise_deployment.dot', 'w') as f:
        f.write(layer_wise_dag.source)
    
    with open('/home/wzc/data/file-share/2025-09-03-14-08-07/baseline_tensor_pipeline.dot', 'w') as f:
        f.write(baseline_dag.source)
    
    print("All DAGs generated successfully!")
    print("Generated files:")
    print("- layer_wise_deployment.svg")
    print("- baseline_tensor_pipeline.svg")
    print("- detailed layer DAGs for layers 0, 7, 15")
    print("- All corresponding .dot files")