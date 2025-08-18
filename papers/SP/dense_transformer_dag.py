import graphviz

# Create a directed graph
dot = graphviz.Digraph(comment='Dense Transformer with Ring Attention + Sequence Parallelism on 16 GPUs')
dot.attr(rankdir='TB', size='50,50')

# Define colors for different GPUs
colors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD',
    '#00D2D3', '#FF9F43', '#10AC84', '#EE5A24', '#0652DD', '#9980FA', '#D63031', '#74B9FF'
]

# Input layer - split across 16 GPUs
for gpu_id in range(16):
    with dot.subgraph(name=f'cluster_input_{gpu_id}') as c:
        c.attr(label=f'GPU {gpu_id}', style='filled', color=colors[gpu_id], fillcolor=colors[gpu_id] + '20')
        
        # Input embedding split by sequence length
        c.node(f'input_{gpu_id}', f'Input Embedding\nX[{gpu_id}] ∈ ℝ^(B×L/16×d_model)', 
               shape='box', style='filled', fillcolor=colors[gpu_id])

# Layer 1 components
for layer in range(1, 5):
    for gpu_id in range(16):
        with dot.subgraph(name=f'cluster_layer{layer}_gpu{gpu_id}') as c:
            c.attr(label=f'Layer {layer} - GPU {gpu_id}', style='filled', color=colors[gpu_id], fillcolor=colors[gpu_id] + '20')
            
            # LayerNorm 1
            c.node(f'ln1_{layer}_{gpu_id}', f'LayerNorm 1\nℝ^(B×L/16×d_model)', 
                   shape='ellipse', style='filled', fillcolor=colors[gpu_id])
            
            # Q, K, V projections (Column Parallel)
            c.node(f'q_proj_{layer}_{gpu_id}', f'Q Projection (Column)\nW_Q ∈ ℝ^(d_model×d_h×H)\nℝ^(B×L/16×d_h×H)', 
                   shape='box', style='filled', fillcolor=colors[gpu_id])
            c.node(f'k_proj_{layer}_{gpu_id}', f'K Projection (Column)\nW_K ∈ ℝ^(d_model×d_h×H)\nℝ^(B×L/16×d_h×H)', 
                   shape='box', style='filled', fillcolor=colors[gpu_id])
            c.node(f'v_proj_{layer}_{gpu_id}', f'V Projection (Column)\nW_V ∈ ℝ^(d_model×d_h×H)\nℝ^(B×L/16×d_h×H)', 
                   shape='box', style='filled', fillcolor=colors[gpu_id])
            
            # Ring Attention computation
            for stage in range(16):
                c.node(f'ring_attn_{layer}_{gpu_id}_stage{stage}', 
                       f'Ring Attention Stage {stage}\nQ[{gpu_id}]×K[{(gpu_id-stage)%16}]×V[{(gpu_id-stage)%16}]\nℝ^(B×L/16×d_h×H)', 
                       shape='diamond', style='filled', fillcolor=colors[gpu_id])
            
            # Attention output projection (Row Parallel)
            c.node(f'o_proj_{layer}_{gpu_id}', f'O Projection (Row)\nW_O ∈ ℝ^(d_h×H×d_model)\nℝ^(B×L/16×d_model)', 
                   shape='box', style='filled', fillcolor=colors[gpu_id])
            
            # Residual connection 1
            c.node(f'residual1_{layer}_{gpu_id}', f'Residual 1\nℝ^(B×L/16×d_model)', 
                   shape='ellipse', style='filled', fillcolor=colors[gpu_id])
            
            # LayerNorm 2
            c.node(f'ln2_{layer}_{gpu_id}', f'LayerNorm 2\nℝ^(B×L/16×d_model)', 
                   shape='ellipse', style='filled', fillcolor=colors[gpu_id])
            
            # FFN components
            c.node(f'ffn_up_{layer}_{gpu_id}', f'FFN Up (Column)\nW_up ∈ ℝ^(d_model×4d_model)\nℝ^(B×L/16×4d_model)', 
                   shape='box', style='filled', fillcolor=colors[gpu_id])
            c.node(f'ffn_act_{layer}_{gpu_id}', f'Activation\nℝ^(B×L/16×4d_model)', 
                   shape='ellipse', style='filled', fillcolor=colors[gpu_id])
            c.node(f'ffn_down_{layer}_{gpu_id}', f'FFN Down (Row)\nW_down ∈ ℝ^(4d_model×d_model)\nℝ^(B×L/16×d_model)', 
                   shape='box', style='filled', fillcolor=colors[gpu_id])
            
            # Residual connection 2
            c.node(f'residual2_{layer}_{gpu_id}', f'Residual 2\nℝ^(B×L/16×d_model)', 
                   shape='ellipse', style='filled', fillcolor=colors[gpu_id])

# Output layer
for gpu_id in range(16):
    with dot.subgraph(name=f'cluster_output_{gpu_id}') as c:
        c.attr(label=f'Output - GPU {gpu_id}', style='filled', color=colors[gpu_id], fillcolor=colors[gpu_id] + '20')
        
        # Final LayerNorm
        c.node(f'final_ln_{gpu_id}', f'Final LayerNorm\nℝ^(B×L/16×d_model)', 
               shape='ellipse', style='filled', fillcolor=colors[gpu_id])
        
        # Output projection
        c.node(f'output_proj_{gpu_id}', f'Output Projection\nℝ^(B×L/16×vocab_size)', 
               shape='box', style='filled', fillcolor=colors[gpu_id])

# Connect the layers
# Input to Layer 1
for gpu_id in range(16):
    dot.edge(f'input_{gpu_id}', f'ln1_1_{gpu_id}')

# Layer connections
for layer in range(1, 5):
    for gpu_id in range(16):
        # LayerNorm 1 to projections
        dot.edge(f'ln1_{layer}_{gpu_id}', f'q_proj_{layer}_{gpu_id}')
        dot.edge(f'ln1_{layer}_{gpu_id}', f'k_proj_{layer}_{gpu_id}')
        dot.edge(f'ln1_{layer}_{gpu_id}', f'v_proj_{layer}_{gpu_id}')
        
        # Projections to Ring Attention
        dot.edge(f'q_proj_{layer}_{gpu_id}', f'ring_attn_{layer}_{gpu_id}_stage0')
        dot.edge(f'k_proj_{layer}_{gpu_id}', f'ring_attn_{layer}_{gpu_id}_stage0')
        dot.edge(f'v_proj_{layer}_{gpu_id}', f'ring_attn_{layer}_{gpu_id}_stage0')
        
        # Ring Attention stages
        for stage in range(15):
            dot.edge(f'ring_attn_{layer}_{gpu_id}_stage{stage}', f'ring_attn_{layer}_{gpu_id}_stage{stage+1}')
            
            # Communication between GPUs for KV blocks
            src_gpu = (gpu_id - stage - 1) % 16
            dst_gpu = (gpu_id - stage) % 16
            dot.edge(f'k_proj_{layer}_{src_gpu}', f'ring_attn_{layer}_{dst_gpu}_stage{stage+1}', 
                     label=f'KV Block Transfer\nK