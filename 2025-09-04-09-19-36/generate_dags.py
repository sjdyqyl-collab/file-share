import os
from graphviz import Digraph

def create_baseline_dag():
    """Create baseline DAG with TP=8, PP=2"""
    dot = Digraph(comment='Baseline DAG: TP=8, PP=2', format='svg')
    dot.attr(rankdir='TB', size='20,30')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input\n[1024, 8192]\nAll GPUs', shape='ellipse', fillcolor='lightgreen')
    
    # Pipeline stage 0 (layers 1-8)
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0\n(GPUs 0-7)', style='dashed', color='red')
        
        # Tensor parallel group 0
        with stage0.subgraph(name='cluster_tp0') as tp0:
            tp0.attr(label='Tensor Parallel Group 0', style='dotted', color='blue')
            
            # Layer 1
            tp0.node('l1_attn_0', 'Layer 1\nMulti-Head Attention\n[1024, 8192] -> [1024, 8192]\nGPU 0', shape='rectangle')
            tp0.node('l1_ffn_0', 'Layer 1\nFFN\n[1024, 8192] -> [1024, 32768] -> [1024, 8192]\nGPU 0', shape='rectangle')
            tp0.node('l1_add_0', 'Layer 1\nResidual Add\n[1024, 8192]\nGPU 0', shape='parallelogram')
            
            tp0.node('l1_attn_1', 'Layer 1\nMulti-Head Attention\n[1024, 8192] -> [1024, 8192]\nGPU 1', shape='rectangle')
            tp0.node('l1_ffn_1', 'Layer 1\nFFN\n[1024, 8192] -> [1024, 32768] -> [1024, 8192]\nGPU 1', shape='rectangle')
            tp0.node('l1_add_1', 'Layer 1\nResidual Add\n[1024, 8192]\nGPU 1', shape='parallelogram')
            
            # ... (similar for GPUs 2-7)
            for i in range(2, 8):
                tp0.node(f'l1_attn_{i}', f'Layer 1\nMulti-Head Attention\n[1024, 8192] -> [1024, 8192]\nGPU {i}', shape='rectangle')
                tp0.node(f'l1_ffn_{i}', f'Layer 1\nFFN\n[1024, 8192] -> [1024, 32768] -> [1024, 8192]\nGPU {i}', shape='rectangle')
                tp0.node(f'l1_add_{i}', f'Layer 1\nResidual Add\n[1024, 8192]\nGPU {i}', shape='parallelogram')
            
            # Add aggregation nodes for tensor parallel
            tp0.node('l1_attn_agg', 'Attention\nAll-Reduce\n[1024, 8192]\nAll GPUs 0-7', shape='ellipse', fillcolor='yellow')
            tp0.node('l1_ffn_agg', 'FFN\nAll-Reduce\n[1024, 8192]\nAll GPUs 0-7', shape='ellipse', fillcolor='yellow')
    
    # Similar structure for layers 2-8 in stage 0
    for layer in range(2, 9):
        with stage0.subgraph(name=f'cluster_tp{layer-1}') as tp:
            tp.attr(label=f'Tensor Parallel Group {layer-1}', style='dotted', color='blue')
            
            for gpu in range(8):
                tp.node(f'l{layer}_attn_{gpu}', f'Layer {layer}\nMulti-Head Attention\n[1024, 8192] -> [1024, 8192]\nGPU {gpu}', shape='rectangle')
                tp.node(f'l{layer}_ffn_{gpu}', f'Layer {layer}\nFFN\n[1024, 8192] -> [1024, 32768] -> [1024, 8192]\nGPU {gpu}', shape='rectangle')
                tp.node(f'l{layer}_add_{gpu}', f'Layer {layer}\nResidual Add\n[1024, 8192]\nGPU {gpu}', shape='parallelogram')
            
            tp.node(f'l{layer}_attn_agg', f'Attention\nAll-Reduce\n[1024, 8192]\nAll GPUs 0-7', shape='ellipse', fillcolor='yellow')
            tp.node(f'l{layer}_ffn_agg', f'FFN\nAll-Reduce\n[1024, 8192]\nAll GPUs 0-7', shape='ellipse', fillcolor='yellow')
    
    # Pipeline stage 1 (layers 9-16)
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1\n(GPUs 8-15)', style='dashed', color='red')
        
        for layer in range(9, 17):
            with stage1.subgraph(name=f'cluster_tp{layer-1}') as tp:
                tp.attr(label=f'Tensor Parallel Group {layer-1}', style='dotted', color='blue')
                
                for gpu in range(8, 16):
                    tp.node(f'l{layer}_attn_{gpu}', f'Layer {layer}\nMulti-Head Attention\n[1024, 8192] -> [1024, 8192]\nGPU {gpu}', shape='rectangle')
                    tp.node(f'l{layer}_ffn_{gpu}', f'Layer {layer}\nFFN\n[1024, 8192] -> [1024, 32768] -> [1024, 8192]\nGPU {gpu}', shape='rectangle')
                    tp.node(f'l{layer}_add_{gpu}', f'Layer {layer}\nResidual Add\n[1024, 8192]\nGPU {gpu}', shape='parallelogram')
                
                tp.node(f'l{layer}_attn_agg', f'Attention\nAll-Reduce\n[1024, 8192]\nAll GPUs 8-15', shape='ellipse', fillcolor='yellow')
                tp.node(f'l{layer}_ffn_agg', f'FFN\nAll-Reduce\n[1024, 8192]\nAll GPUs 8-15', shape='ellipse', fillcolor='yellow')
    
    # Pipeline communication
    dot.node('pipe_comm', 'Pipeline\nCommunication\n[1024, 8192]\nStage 0 -> Stage 1', shape='ellipse', fillcolor='orange')
    
    # Output node
    dot.node('output', 'Output\n[1024, 8192]\nAll GPUs', shape='ellipse', fillcolor='lightgreen')
    
    # Connect nodes
    # Input to first layer
    dot.edge('input', 'l1_attn_0')
    dot.edge('input', 'l1_attn_1')
    for i in range(2, 8):
        dot.edge('input', f'l1_attn_{i}')
    
    # Layer 1 connections
    for gpu in range(8):
        dot.edge(f'l{1}_attn_{gpu}', f'l{1}_ffn_{gpu}')
        dot.edge(f'l{1}_ffn_{gpu}', f'l{1}_add_{gpu}')
        dot.edge(f'l{1}_attn_{gpu}', f'l{1}_add_{gpu}')  # Residual
    
    # Connect to aggregation
    for gpu in range(8):
        dot.edge(f'l{1}_add_{gpu}', 'l1_attn_agg')
        dot.edge(f'l1_attn_agg', f'l{1}_ffn_agg')
    
    # Similar connections for layers 2-8
    for layer in range(2, 9):
        for gpu in range(8):
            dot.edge(f'l{layer-1}_ffn_agg', f'l{layer}_attn_{gpu}')
            dot.edge(f'l{layer}_attn_{gpu}', f'l{layer}_ffn_{gpu}')
            dot.edge(f'l{layer}_ffn_{gpu}', f'l{layer}_add_{gpu}')
            dot.edge(f'l{layer}_attn_{gpu}', f'l{layer}_add_{gpu}')
            dot.edge(f'l{layer}_add_{gpu}', f'l{layer}_attn_agg')
            dot.edge(f'l{layer}_attn_agg', f'l{layer}_ffn_agg')
    
    # Pipeline communication
    dot.edge('l8_ffn_agg', 'pipe_comm')
    
    # Stage 1 connections
    for gpu in range(8, 16):
        dot.edge('pipe_comm', f'l9_attn_{gpu}')
    
    for layer in range(9, 17):
        for gpu in range(8, 16):
            dot.edge(f'l{layer}_attn_{gpu}', f'l{layer}_ffn_{gpu}')
            dot.edge(f'l{layer}_ffn_{gpu}', f'l{layer}_add_{gpu}')
            dot.edge(f'l{layer}_attn_{gpu}', f'l{layer}_add_{gpu}')
            dot.edge(f'l{layer}_add_{gpu}', f'l{layer}_attn_agg')
            dot.edge(f'l{layer}_attn_agg', f'l{layer}_ffn_agg')
    
    # Final output
    dot.edge('l16_ffn_agg', 'output')
    
    return dot

def create_proposed_dag():
    """Create proposed DAG with layer-wise deployment (1 layer per GPU)"""
    dot = Digraph(comment='Proposed DAG: Layer-wise Deployment', format='svg')
    dot.attr(rankdir='TB', size='20,30')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input\n[1024, 8192]\nGPU 0', shape='ellipse', fillcolor='lightgreen')
    
    # Create 16 layers, each on a separate GPU
    layers = []
    for layer in range(1, 17):
        gpu_id = layer - 1
        
        # Create subgraph for each layer
        with dot.subgraph(name=f'cluster_layer{layer}') as cluster:
            cluster.attr(label=f'Layer {layer} (GPU {gpu_id})', style='dashed', color='purple')
            
            # Multi-Head Attention
            cluster.node(f'l{layer}_attn', f'Multi-Head Attention\n16 heads × 512 dim\n[1024, 8192] -> [1024, 8192]\nGPU {gpu_id}', 
                        shape='rectangle', fillcolor='lightcoral')
            
            # Attention residual add
            cluster.node(f'l{layer}_attn_add', f'Residual Add\n[1024, 8192]\nGPU {gpu_id}', 
                        shape='parallelogram', fillcolor='lightyellow')
            
            # Layer Norm after attention
            cluster.node(f'l{layer}_attn_norm', f'Layer Norm\n[1024, 8192]\nGPU {gpu_id}', 
                        shape='rectangle', fillcolor='lightpink')
            
            # FFN
            cluster.node(f'l{layer}_ffn1', f'FFN Linear 1\n[1024, 8192] -> [1024, 32768]\nGPU {gpu_id}', 
                        shape='rectangle', fillcolor='lightgreen')
            
            cluster.node(f'l{layer}_gelu', f'GELU Activation\n[1024, 32768]\nGPU {gpu_id}', 
                        shape='rectangle', fillcolor='lightblue')
            
            cluster.node(f'l{layer}_ffn2', f'FFN Linear 2\n[1024, 32768] -> [1024, 8192]\nGPU {gpu_id}', 
                        shape='rectangle', fillcolor='lightgreen')
            
            # FFN residual add
            cluster.node(f'l{layer}_ffn_add', f'Residual Add\n[1024, 8192]\nGPU {gpu_id}', 
                        shape='parallelogram', fillcolor='lightyellow')
            
            # Layer Norm after FFN
            cluster.node(f'l{layer}_ffn_norm', f'Layer Norm\n[1024, 8192]\nGPU {gpu_id}', 
                        shape='rectangle', fillcolor='lightpink')
    
    # Communication nodes between layers
    for layer in range(1, 16):
        dot.node(f'comm_{layer}_{layer+1}', f'Layer Transfer\n[1024, 8192]\nGPU {layer-1} -> GPU {layer}', 
                shape='ellipse', fillcolor='orange')
    
    # Output node
    dot.node('output', 'Output\n[1024, 8192]\nGPU 15', shape='ellipse', fillcolor='lightgreen')
    
    # Connect nodes
    # Input to layer 1
    dot.edge('input', 'l1_attn')
    
    # Layer 1 internal connections
    dot.edge('l1_attn', 'l1_attn_add')
    dot.edge('input', 'l1_attn_add')  # Residual connection
    dot.edge('l1_attn_add', 'l1_attn_norm')
    dot.edge('l1_attn_norm', 'l1_ffn1')
    dot.edge('l1_ffn1', 'l1_gelu')
    dot.edge('l1_gelu', 'l1_ffn2')
    dot.edge('l1_ffn2', 'l1_ffn_add')
    dot.edge('l1_attn_norm', 'l1_ffn_add')  # Residual connection
    dot.edge('l1_ffn_add', 'l1_ffn_norm')
    
    # Connect layers through communication nodes
    for layer in range(1, 16):
        dot.edge(f'l{layer}_ffn_norm', f'comm_{layer}_{layer+1}')
        dot.edge(f'comm_{layer}_{layer+1}', f'l{layer+1}_attn')
    
    # Similar internal connections for all layers
    for layer in range(2, 17):
        # Create internal connections for each layer
        prev_output = f'l{layer-1}_ffn_norm' if layer > 1 else 'input'
        
        # Internal layer connections
        dot.edge(f'l{layer}_attn', f'l{layer}_attn_add')
        dot.edge(f'l{layer-1}_ffn_norm', f'l{layer}_attn_add')  # Via communication
        dot.edge(f'l{layer}_attn_add', f'l{layer}_attn_norm')
        dot.edge(f'l{layer}_attn_norm', f'l{layer}_ffn1')
        dot.edge(f'l{layer}_ffn1', f'l{layer}_gelu')
        dot.edge(f'l{layer}_gelu', f'l{layer}_ffn2')
        dot.edge(f'l{layer}_ffn2', f'l{layer}_ffn_add')
        dot.edge(f'l{layer}_attn_norm', f'l{layer}_ffn_add')  # Residual connection
        dot.edge(f'l{layer}_ffn_add', f'l{layer}_ffn_norm')
    
    # Final output
    dot.edge('l16_ffn_norm', 'output')
    
    return dot

if __name__ == '__main__':
    # Create baseline DAG
    baseline_dag = create_baseline_dag()
    baseline_dag.render('/home/wzc/data/file-share/2025-09-04-09-19-36/baseline_dag', format='svg')
    baseline_dag.save('/home/wzc/data/file-share/2025-09-04-09-19-36/baseline_dag.dot')
    
    # Create proposed DAG
    proposed_dag = create_proposed_dag()
    proposed_dag.render('/home/wzc/data/file-share/2025-09-04-09-19-36/proposed_dag', format='svg')
    proposed_dag.save('/home/wzc/data/file-share/2025-09-04-09-19-36/proposed_dag.dot')
    
    print("DAGs generated successfully!")
    print("Files created:")
    print("- baseline_dag.svg")
    print("- baseline_dag.dot")
    print("- proposed_dag.svg")
    print("- proposed_dag.dot")