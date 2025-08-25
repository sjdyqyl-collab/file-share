import graphviz

# Create baseline DAG for Dense Transformer with TP=8, PP=2
dot = graphviz.Digraph(comment='Dense Transformer Baseline (TP=8, PP=2)', format='svg')
dot.attr(rankdir='TB', size='20,30')

# Define node styles
dot.attr('node', shape='ellipse', style='filled', color='lightblue')  # Communication
dot.attr('node', shape='rectangle', style='filled', color='lightgreen')  # Computation
dot.attr('node', shape='parallelogram', style='filled', color='lightyellow')  # Routing/Aggregation

# Input
with dot.subgraph(name='cluster_input') as c:
    c.attr(label='Input Layer')
    c.node('input', 'Input\nX: [B, L, d_model]\nGPU: Host', shape='parallelogram')

# Pipeline Stage 0 (Layers 0-1) - GPUs 0-7
with dot.subgraph(name='cluster_stage0') as c:
    c.attr(label='Pipeline Stage 0 (GPUs 0-7)')
    
    # Layer 0
    with dot.subgraph(name='cluster_layer0') as layer:
        layer.attr(label='Layer 0')
        
        # Input split for pipeline
        layer.node('split0', 'Split for Pipeline\nX: [B, L/P, d_model]\nGPU: 0-7', shape='parallelogram')
        
        # MHA - Tensor Parallel across 8 GPUs
        with dot.subgraph(name='cluster_mha0') as mha:
            mha.attr(label='Multi-Head Attention')
            
            # QKV projections (column parallel)
            for i in range(8):
                mha.node(f'qkv0_{i}', f'QKV Projection {i}\n[Q,K,V]: [B, L/P, d_model/8]\nGPU: {i}', shape='rectangle')
            
            # Attention computation
            for i in range(8):
                mha.node(f'attn0_{i}', f'Attention {i}\nOutput: [B, L/P, d_model/8]\nGPU: {i}', shape='rectangle')
            
            # Output projection (row parallel)
            for i in range(8):
                mha.node(f'out0_{i}', f'Output Proj {i}\nOutput: [B, L/P, d_model/8]\nGPU: {i}', shape='rectangle')
            
            # All-reduce for attention output
            mha.node('ar0', 'All-Reduce\nSum across 8 GPUs\nGPU: 0-7', shape='ellipse')
        
        # Residual connection
        layer.node('res0', 'Residual Add\nInput: [B, L/P, d_model]\nGPU: 0-7', shape='parallelogram')
        
        # FFN - Tensor Parallel across 8 GPUs
        with dot.subgraph(name='cluster_ffn0') as ffn:
            ffn.attr(label='Feed Forward Network')
            
            # First linear (column parallel)
            for i in range(8):
                ffn.node(f'ffn1_0_{i}', f'FFN Linear1 {i}\nOutput: [B, L/P, ffn_dim/8]\nGPU: {i}', shape='rectangle')
            
            # Activation
            for i in range(8):
                ffn.node(f'act0_{i}', f'GELU {i}\nOutput: [B, L/P, ffn_dim/8]\nGPU: {i}', shape='rectangle')
            
            # Second linear (row parallel)
            for i in range(8):
                ffn.node(f'ffn2_0_{i}', f'FFN Linear2 {i}\nOutput: [B, L/P, d_model/8]\nGPU: {i}', shape='rectangle')
            
            # All-reduce for FFN output
            ffn.node('ar_ffn0', 'All-Reduce\nSum across 8 GPUs\nGPU: 0-7', shape='ellipse')
        
        # Residual connection
        layer.node('res_ffn0', 'Residual Add\nInput: [B, L/P, d_model]\nGPU: 0-7', shape='parallelogram')

# Pipeline communication between stages
dot.node('send_stage0', 'Send to Stage 1\nX: [B, L/P, d_model]\nGPU: 0-7 → 8-15', shape='ellipse')

# Pipeline Stage 1 (Layers 2-3) - GPUs 8-15
with dot.subgraph(name='cluster_stage1') as c:
    c.attr(label='Pipeline Stage 1 (GPUs 8-15)')
    
    # Layer 2
    with dot.subgraph(name='cluster_layer2') as layer:
        layer.attr(label='Layer 2')
        
        # Receive from stage 0
        layer.node('recv_stage1', 'Receive from Stage 0\nX: [B, L/P, d_model]\nGPU: 8-15', shape='ellipse')
        
        # MHA - Tensor Parallel across 8 GPUs (8-15)
        with dot.subgraph(name='cluster_mha2') as mha:
            mha.attr(label='Multi-Head Attention')
            
            # QKV projections (column parallel)
            for i in range(8, 16):
                mha.node(f'qkv2_{i}', f'QKV Projection {i}\n[Q,K,V]: [B, L/P, d_model/8]\nGPU: {i}', shape='rectangle')
            
            # Attention computation
            for i in range(8, 16):
                mha.node(f'attn2_{i}', f'Attention {i}\nOutput: [B, L/P, d_model/8]\nGPU: {i}', shape='rectangle')
            
            # Output projection (row parallel)
            for i in range(8, 16):
                mha.node(f'out2_{i}', f'Output Proj {i}\nOutput: [B, L/P, d_model/8]\nGPU: {i}', shape='rectangle')
            
            # All-reduce for attention output
            mha.node('ar2', 'All-Reduce\nSum across 8 GPUs\nGPU: 8-15', shape='ellipse')
        
        # Residual connection
        layer.node('res2', 'Residual Add\nInput: [B, L/P, d_model]\nGPU: 8-15', shape='parallelogram')
        
        # FFN - Tensor Parallel across 8 GPUs
        with dot.subgraph(name='cluster_ffn2') as ffn:
            ffn.attr(label='Feed Forward Network')
            
            # First linear (column parallel)
            for i in range(8, 16):
                ffn.node(f'ffn1_2_{i}', f'FFN Linear1 {i}\nOutput: [B, L/P, ffn_dim/8]\nGPU: {i}', shape='rectangle')
            
            # Activation
            for i in range(8, 16):
                ffn.node(f'act2_{i}', f'GELU {i}\nOutput: [B, L/P, ffn_dim/8]\nGPU: {i}', shape='rectangle')
            
            # Second linear (row parallel)
            for i in range(8, 16):
                ffn.node(f'ffn2_2_{i}', f'FFN Linear2 {i}\nOutput: [B, L/P, d_model/8]\nGPU: {i}', shape='rectangle')
            
            # All-reduce for FFN output
            ffn.node('ar_ffn2', 'All-Reduce\nSum across 8 GPUs\nGPU: 8-15', shape='ellipse')
        
        # Residual connection
        layer.node('res_ffn2', 'Residual Add\nInput: [B, L/P, d_model]\nGPU: 8-15', shape='parallelogram')

    # Layer 3 (similar structure)
    with dot.subgraph(name='cluster_layer3') as layer:
        layer.attr(label='Layer 3')
        
        # MHA - Tensor Parallel across 8 GPUs (8-15)
        with dot.subgraph(name='cluster_mha3') as mha:
            mha.attr(label='Multi-Head Attention')
            
            # QKV projections (column parallel)
            for i in range(8, 16):
                mha.node(f'qkv3_{i}', f'QKV Projection {i}\n[Q,K,V]: [B, L/P, d_model/8]\nGPU: {i}', shape='rectangle')
            
            # Attention computation
            for i in range(8, 16):
                mha.node(f'attn3_{i}', f'Attention {i}\nOutput: [B, L/P, d_model/8]\nGPU: {i}', shape='rectangle')
            
            # Output projection (row parallel)
            for i in range(8, 16):
                mha.node(f'out3_{i}', f'Output Proj {i}\nOutput: [B, L/P, d_model/8]\nGPU: {i}', shape='rectangle')
            
            # All-reduce for attention output
            mha.node('ar3', 'All-Reduce\nSum across 8 GPUs\nGPU: 8-15', shape='ellipse')
        
        # Residual connection
        layer.node('res3', 'Residual Add\nInput: [B, L/P, d_model]\nGPU: 8-15', shape='parallelogram')
        
        # FFN - Tensor Parallel across 8 GPUs
        with dot.subgraph(name='cluster_ffn3') as ffn:
            ffn.attr(label='Feed Forward Network')
            
            # First linear (column parallel)
            for i in range(8, 16):
                ffn.node(f'ffn1_3_{i}', f'FFN Linear1 {i}\nOutput: [B, L/P, ffn_dim/8]\nGPU: {i}', shape='rectangle')
            
            # Activation
            for i in range(8, 16):
                ffn.node(f'act3_{i}', f'GELU {i}\nOutput: [B, L/P, ffn_dim/8]\nGPU: {i}', shape='rectangle')
            
            # Second linear (row parallel)
            for i in range(8, 16):
                ffn.node(f'ffn2_3_{i}', f'FFN Linear2 {i}\nOutput: [B, L/P, d_model/8]\nGPU: {i}', shape='rectangle')
            
            # All-reduce for FFN output
            ffn.node('ar_ffn3', 'All-Reduce\nSum across 8 GPUs\nGPU: 8-15', shape='ellipse')
        
        # Residual connection
        layer.node('res_ffn3', 'Residual Add\nInput: [B, L/P, d_model]\nGPU: 8-15', shape='parallelogram')

# Output
with dot.subgraph(name='cluster_output') as c:
    c.attr(label='Output Layer')
    c.node('gather', 'Gather from Pipeline\nX: [B, L, d_model]\nGPU: 8-15 → Host', shape='parallelogram')
    c.node('output', 'Output\nX: [B, L, d_model]\nGPU: Host', shape='parallelogram')

# Connections
# Input to split
dot.edge('input', 'split0')

# Layer 0 connections
for i in range(8):
    dot.edge('split0', f'qkv0_{i}')
    dot.edge(f'qkv0_{i}', f'attn0_{i}')
    dot.edge(f'attn0_{i}', f'out0_{i}')
    dot.edge(f'out0_{i}', 'ar0')

dot.edge('ar0', 'res0')
dot.edge('split0', 'res0')  # Residual connection

# Layer 0 FFN connections
for i in range(8):
    dot.edge('res0', f'ffn1_0_{i}')
    dot.edge(f'ffn1_0_{i}', f'act0_{i}')
    dot.edge(f'act0_{i}', f'ffn2_0_{i}')
    dot.edge(f'ffn2_0_{i}', 'ar_ffn0')

dot.edge('ar_ffn0', 'res_ffn0')
dot.edge('res0', 'res_ffn0')  # Residual connection

# Pipeline communication
dot.edge('res_ffn0', 'send_stage0')
dot.edge('send_stage0', 'recv_stage1')

# Layer 2 connections
for i in range(8, 16):
    dot.edge('recv_stage1', f'qkv2_{i}')
    dot.edge(f'qkv2_{i}', f'attn2_{i}')
    dot.edge(f'attn2_{i}', f'out2_{i}')
    dot.edge(f'out2_{i}', 'ar2')

dot.edge('ar2', 'res2')
dot.edge('recv_stage1', 'res2')  # Residual connection

# Layer 2 FFN connections
for i in range(8, 16):
    dot.edge('res2', f'ffn1_2_{i}')
    dot.edge(f'ffn1_2_{i}', f'act2_{i}')
    dot.edge(f'act2_{i}', f'ffn2_2_{i}')
    dot.edge(f'ffn2_2_{i}', 'ar_ffn2')

dot.edge('ar_ffn2', 'res_ffn2')
dot.edge('res2', 'res_ffn2')  # Residual connection

# Layer 3 connections
for i in range(8, 16):
    dot.edge('res_ffn2', f'qkv3_{i}')
    dot.edge(f'qkv3_{i}', f'attn3_{i}')
    dot.edge(f'attn3_{i}', f'out3_{i}')
    dot.edge(f'out3_{i}', 'ar3')

dot.edge('ar3', 'res3')
dot.edge('res_ffn2', 'res3')  # Residual connection

# Layer 3 FFN connections
for i in range(8, 16):
    dot.edge('res3', f'ffn1_3_{i}')
    dot.edge(f'ffn1_3_{i}', f'act3_{i}')
    dot.edge(f'act3_{i}', f'ffn2_3_{i}')
    dot.edge(f'ffn2_3_{i}', 'ar_ffn3')

dot.edge('ar_ffn3', 'res_ffn3')
dot.edge('res3', 'res_ffn3')  # Residual connection

# Output connections
dot.edge('res_ffn3', 'gather')
dot.edge('gather', 'output')

# Save the DAG
dot.render('/home/wzc/data/file-share/submission/baseline_dense_dag', format='svg')
print("Baseline Dense DAG saved to /home/wzc/data/file-share/submission/baseline_dense_dag.svg")