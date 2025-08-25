import graphviz

# Create baseline DAG for MoE Transformer with TP=8, PP=2
dot = graphviz.Digraph(comment='MoE Transformer Baseline (TP=8, PP=2)', format='svg')
dot.attr(rankdir='TB', size='20,30')

# Define node styles
dot.attr('node', shape='ellipse', style='filled', color='lightblue')  # Communication
dot.attr('node', shape='rectangle', style='filled', color='lightgreen')  # Computation
dot.attr('node', shape='parallelogram', style='filled', color='lightyellow')  # Routing/Aggregation
dot.attr('node', shape='diamond', style='filled', color='lightcoral')  # Gate

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
        
        # MoE - Tensor Parallel across 8 GPUs with 8 experts
        with dot.subgraph(name='cluster_moe0') as moe:
            moe.attr(label='Mixture of Experts')
            
            # Gate computation
            moe.node('gate0', 'Gate\nCompute routing scores\nInput: [B, L/P, d_model]\nGPU: 0-7', shape='diamond')
            
            # Expert selection (top-2)
            moe.node('select0', 'Select Top-2 Experts\nGPU: 0-7', shape='parallelogram')
            
            # Expert computation (8 experts distributed)
            for exp_id in range(8):
                with dot.subgraph(name=f'cluster_expert0_{exp_id}') as expert:
                    expert.attr(label=f'Expert {exp_id}')
                    
                    # Expert linear layers
                    expert.node(f'expert0_{exp_id}_1', f'Expert {exp_id} Linear1\nOutput: [B, L/P, ffn_dim]\nGPU: {exp_id}', shape='rectangle')
                    expert.node(f'expert0_{exp_id}_act', f'Expert {exp_id} GELU\nGPU: {exp_id}', shape='rectangle')
                    expert.node(f'expert0_{exp_id}_2', f'Expert {exp_id} Linear2\nOutput: [B, L/P, d_model]\nGPU: {exp_id}', shape='rectangle')
            
            # Combine expert outputs
            moe.node('combine0', 'Combine Expert Outputs\nWeighted sum by gate scores\nGPU: 0-7', shape='parallelogram')
        
        # Residual connection
        layer.node('res_moe0', 'Residual Add\nInput: [B, L/P, d_model]\nGPU: 0-7', shape='parallelogram')

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
        
        # MoE - Tensor Parallel across 8 GPUs with 8 experts
        with dot.subgraph(name='cluster_moe2') as moe:
            moe.attr(label='Mixture of Experts')
            
            # Gate computation
            moe.node('gate2', 'Gate\nCompute routing scores\nInput: [B, L/P, d_model]\nGPU: 8-15', shape='diamond')
            
            # Expert selection (top-2)
            moe.node('select2', 'Select Top-2 Experts\nGPU: 8-15', shape='parallelogram')
            
            # Expert computation (8 experts distributed)
            for exp_id in range(8):
                with dot.subgraph(name=f'cluster_expert2_{exp_id}') as expert:
                    expert.attr(label=f'Expert {exp_id+8}')
                    
                    # Expert linear layers
                    expert.node(f'expert2_{exp_id}_1', f'Expert {exp_id+8} Linear1\nOutput: [B, L/P, ffn_dim]\nGPU: {exp_id+8}', shape='rectangle')
                    expert.node(f'expert2_{exp_id}_act', f'Expert {exp_id+8} GELU\nGPU: {exp_id+8}', shape='rectangle')
                    expert.node(f'expert2_{exp_id}_2', f'Expert {exp_id+8} Linear2\nOutput: [B, L/P, d_model]\nGPU: {exp_id+8}', shape='rectangle')
            
            # Combine expert outputs
            moe.node('combine2', 'Combine Expert Outputs\nWeighted sum by gate scores\nGPU: 8-15', shape='parallelogram')
        
        # Residual connection
        layer.node('res_moe2', 'Residual Add\nInput: [B, L/P, d_model]\nGPU: 8-15', shape='parallelogram')

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
        
        # MoE - Tensor Parallel across 8 GPUs with 8 experts
        with dot.subgraph(name='cluster_moe3') as moe:
            moe.attr(label='Mixture of Experts')
            
            # Gate computation
            moe.node('gate3', 'Gate\nCompute routing scores\nInput: [B, L/P, d_model]\nGPU: 8-15', shape='diamond')
            
            # Expert selection (top-2)
            moe.node('select3', 'Select Top-2 Experts\nGPU: 8-15', shape='parallelogram')
            
            # Expert computation (8 experts distributed)
            for exp_id in range(8):
                with dot.subgraph(name=f'cluster_expert3_{exp_id}') as expert:
                    expert.attr(label=f'Expert {exp_id+8}')
                    
                    # Expert linear layers
                    expert.node(f'expert3_{exp_id}_1', f'Expert {exp_id+8} Linear1\nOutput: [B, L/P, ffn_dim]\nGPU: {exp_id+8}', shape='rectangle')
                    expert.node(f'expert3_{exp_id}_act', f'Expert {exp_id+8} GELU\nGPU: {exp_id+8}', shape='rectangle')
                    expert.node(f'expert3_{exp_id}_2', f'Expert {exp_id+8} Linear2\nOutput: [B, L/P, d_model]\nGPU: {exp_id+8}', shape='rectangle')
            
            # Combine expert outputs
            moe.node('combine3', 'Combine Expert Outputs\nWeighted sum by gate scores\nGPU: 8-15', shape='parallelogram')
        
        # Residual connection
        layer.node('res_moe3', 'Residual Add\nInput: [B, L/P, d_model]\nGPU: 8-15', shape='parallelogram')

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

# Layer 0 MoE connections
dot.edge('res0', 'gate0')
dot.edge('gate0', 'select0')

# Expert connections with routing
for exp_id in range(8):
    dot.edge('select0', f'expert0_{exp_id}_1', style='dashed', label=f'if selected')
    dot.edge(f'expert0_{exp_id}_1', f'expert0_{exp_id}_act')
    dot.edge(f'expert0_{exp_id}_act', f'expert0_{exp_id}_2')
    dot.edge(f'expert0_{exp_id}_2', 'combine0')

dot.edge('combine0', 'res_moe0')
dot.edge('res0', 'res_moe0')  # Residual connection

# Pipeline communication
dot.edge('res_moe0', 'send_stage0')
dot.edge('send_stage0', 'recv_stage1')

# Layer 2 connections
for i in range(8, 16):
    dot.edge('recv_stage1', f'qkv2_{i}')
    dot.edge(f'qkv2_{i}', f'attn2_{i}')
    dot.edge(f'attn2_{i}', f'out2_{i}')
    dot.edge(f'out2_{i}', 'ar2')

dot.edge('ar2', 'res2')
dot.edge('recv_stage1', 'res2')  # Residual connection

# Layer 2 MoE connections
dot.edge('res2', 'gate2')
dot.edge('gate2', 'select2')

# Expert connections with routing
for exp_id in range(8):
    dot.edge('select2', f'expert2_{exp_id}_1', style='dashed', label=f'if selected')
    dot.edge(f'expert2_{exp_id}_1', f'expert2_{exp_id}_act')
    dot.edge(f'expert2_{exp_id}_act', f'expert2_{exp_id}_2')
    dot.edge(f'expert2_{exp_id}_2', 'combine2')

dot.edge('combine2', 'res_moe2')
dot.edge('res2', 'res_moe2')  # Residual connection

# Layer 3 connections
for i in range(8, 16):
    dot.edge('res_moe2', f'qkv3_{i}')
    dot.edge(f'qkv3_{i}', f'attn3_{i}')
    dot.edge(f'attn3_{i}', f'out3_{i}')
    dot.edge(f'out3_{i}', 'ar3')

dot.edge('ar3', 'res3')
dot.edge('res_moe2', 'res3')  # Residual connection

# Layer 3 MoE connections
dot.edge('res3', 'gate3')
dot.edge('gate3', 'select3')

# Expert connections with routing
for exp_id in range(8):
    dot.edge('select3', f'expert3_{exp_id}_1', style='dashed', label=f'if selected')
    dot.edge(f'expert3_{exp_id}_1', f'expert3_{exp_id}_act')
    dot.edge(f'expert3_{exp_id}_act', f'expert3_{exp_id}_2')
    dot.edge(f'expert3_{exp_id}_2', 'combine3')

dot.edge('combine3', 'res_moe3')
dot.edge('res3', 'res_moe3')  # Residual connection

# Output connections
dot.edge('res_moe3', 'gather')
dot.edge('gather', 'output')

# Save the DAG
dot.render('/home/wzc/data/file-share/submission/baseline_moe_dag', format='svg')
print("Baseline MoE DAG saved to /home/wzc/data/file-share/submission/baseline_moe_dag.svg")