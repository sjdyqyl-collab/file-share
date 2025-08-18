import graphviz

# Create a directed graph for MoE transformer
dot = graphviz.Digraph(comment='MoE Transformer with Ring Attention + Sequence Parallelism on 16 GPUs')
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
        c.node(f'input_{gpu_id}', f'Input Embedding\nX[{gpu_id}] ∈ ℝ^(B×L/16×d_model)', 
               shape='box', style='filled', fillcolor=colors[gpu_id])

# Layer 1-4 components
for layer in range(1, 5):
    for gpu_id in range(16):
        with dot.subgraph(name=f'cluster_layer{layer}_gpu{gpu_id}') as c:
            c.attr(label=f'Layer {layer} - GPU {gpu_id}', style='filled', color=colors[gpu_id], fillcolor=colors[gpu_id] + '20')
            
            # MHA components (same as dense)
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
            
            c.node(f'residual1_{layer}_{gpu_id}', f'Residual 1\nℝ^(B×L/16×d_model)', 
                   shape='ellipse', style='filled', fillcolor=colors[gpu_id])
            
            # MoE components
            c.node(f'ln2_{layer}_{gpu_id}', f'LayerNorm 2\nℝ^(B×L/16×d_model)', 
                   shape='ellipse', style='filled', fillcolor=colors[gpu_id])
            
            # Expert routing
            c.node(f'router_{layer}_{gpu_id}', f'Router\nGating Network\nℝ^(B×L/16×8)', 
                   shape='hexagon', style='filled', fillcolor=colors[gpu_id])
            
            # Expert selection (top-2)
            c.node(f'gate_{layer}_{gpu_id}', f'Top-2 Gating\nExpert Selection', 
                   shape='pentagon', style='filled', fillcolor=colors[gpu_id])
            
            # 8 Expert computations (each expert has capacity factor 1.25)
            for expert_id in range(8):
                c.node(f'expert_{layer}_{gpu_id}_{expert_id}', 
                       f'Expert {expert_id}\nW_gate ∈ ℝ^(d_model×4d_model)\nW_up ∈ ℝ^(d_model×4d_model)\nW_down ∈ ℝ^(4d_model×d_model)\nCapacity: 1.25×tokens', 
                       shape='box', style='filled', fillcolor=colors[gpu_id])
            
            # Expert aggregation
            c.node(f'expert_agg_{layer}_{gpu_id}', f'Expert Aggregation\nWeighted Sum\nℝ^(B×L/16×d_model)', 
                   shape='ellipse', style='filled', fillcolor=colors[gpu_id])
            
            c.node(f'residual2_{layer}_{gpu_id}', f'Residual 2\nℝ^(B×L/16×d_model)', 
                   shape='ellipse', style='filled', fillcolor=colors[gpu_id])

# Output layer
for gpu_id in range(16):
    with dot.subgraph(name=f'cluster_output_{gpu_id}') as c:
        c.attr(label=f'Output - GPU {gpu_id}', style='filled', color=colors[gpu_id], fillcolor=colors[gpu_id] + '20')
        c.node(f'final_ln_{gpu_id}', f'Final LayerNorm\nℝ^(B×L/16×d_model)', 
               shape='ellipse', style='filled', fillcolor=colors[gpu_id])
        c.node(f'output_proj_{gpu_id}', f'Output Projection\nℝ^(B×L/16×vocab_size)', 
               shape='box', style='filled', fillcolor=colors[gpu_id])

# Connect the layers
# Input to Layer 1
for gpu_id in range(16):
    dot.edge(f'input_{gpu_id}', f'ln1_1_{gpu_id}')

# Layer connections
for layer in range(1, 5):
    for gpu_id in range(16):
        # MHA connections
        dot.edge(f'ln1_{layer}_{gpu_id}', f'q_proj_{layer}_{gpu_id}')
        dot.edge(f'ln1_{layer}_{gpu_id}', f'k_proj_{layer}_{gpu_id}')
        dot.edge(f'ln1_{layer}_{gpu_id}', f'v_proj_{layer}_{gpu_id}')
        
        dot.edge(f'q_proj_{layer}_{gpu_id}', f'ring_attn_{layer}_{gpu_id}_stage0')
        dot.edge(f'k_proj_{layer}_{gpu_id}', f'ring_attn_{layer}_{gpu_id}_stage0')
        dot.edge(f'v_proj_{layer}_{gpu_id}', f'ring_attn_{layer}_{gpu_id}_stage0')
        
        # Ring Attention stages
        for stage in range(15):
            dot.edge(f'ring_attn_{layer}_{gpu_id}_stage{stage}', f'ring_attn_{layer}_{gpu_id}_stage{stage+1}')
            
            # KV block communication
            src_gpu = (gpu_id - stage - 1) % 16
            dst_gpu = (gpu_id - stage) % 16
            dot.edge(f'k_proj_{layer}_{src_gpu}', f'ring_attn_{layer}_{dst_gpu}_stage{stage+1}', 
                     label=f'KV Block\nK/V[{src_gpu}]→GPU{dst_gpu}', 
                     style='dashed', color='blue', constraint='false')
            dot.edge(f'v_proj_{layer}_{src_gpu}', f'ring_attn_{layer}_{dst_gpu}_stage{stage+1}', 
                     label=f'KV Block\nK/V[{src_gpu}]→GPU{dst_gpu}', 
                     style='dashed', color='blue', constraint='false')
        
        # Attention output path
        dot.edge(f'ring_attn_{layer}_{gpu_id}_stage15', f'o_proj_{layer}_{gpu_id}')
        dot.edge(f'o_proj_{layer}_{gpu_id}', f'residual1_{layer}_{gpu_id}')
        dot.edge(f'ln1_{layer}_{gpu_id}', f'residual1_{layer}_{gpu_id}', style='dotted')
        
        # MoE connections
        dot.edge(f'residual1_{layer}_{gpu_id}', f'ln2_{layer}_{gpu_id}')
        dot.edge(f'ln2_{layer}_{gpu_id}', f'router_{layer}_{gpu_id}')
        dot.edge(f'router_{layer}_{gpu_id}', f'gate_{layer}_{gpu_id}')
        
        # Expert routing and computation
        dot.edge(f'gate_{layer}_{gpu_id}', f'expert_agg_{layer}_{gpu_id}')
        
        # Connect each expert
        for expert_id in range(8):
            dot.edge(f'ln2_{layer}_{gpu_id}', f'expert_{layer}_{gpu_id}_{expert_id}')
            dot.edge(f'expert_{layer}_{gpu_id}_{expert_id}', f'expert_agg_{layer}_{gpu_id}')
            dot.edge(f'gate_{layer}_{gpu_id}', f'expert_{layer}_{gpu_id}_{expert_id}', 
                     label=f'gate weight', style='dotted')
        
        dot.edge(f'expert_agg_{layer}_{gpu_id}', f'residual2_{layer}_{gpu_id}')
        dot.edge(f'ln2_{layer}_{gpu_id}', f'residual2_{layer}_{gpu_id}', style='dotted')

# Connect layers
for layer in range(1, 4):
    for gpu_id in range(16):
        dot.edge(f'residual2_{layer}_{gpu_id}', f'ln1_{layer+1}_{gpu_id}')

# Final output connections
for gpu_id in range(16):
    dot.edge(f'residual2_4_{gpu_id}', f'final_ln_{gpu_id}')
    dot.edge(f'final_ln_{gpu_id}', f'output_proj_{gpu_id}')

# Add communication summary box
with dot.subgraph(name='cluster_communication') as c:
    c.attr(label='Communication Patterns', style='filled', color='lightgray', fillcolor='lightgray')
    c.node('comm_legend', 
           'Ring Attention Communication:\n• KV blocks passed in ring topology\n• Each GPU sends L/16 × d_model per stage\n• 16 stages total per attention\n\nExpert Communication:\n• Expert selection happens locally\n• No cross-GPU expert communication\n• Capacity factor 1.25 for load balancing',
           shape='note', style='filled', fillcolor='white')

# Render the graph
dot.render('/home/wzc/data/papers/SP/moe_transformer_dag', format='dot', cleanup=False)
dot.render('/home/wzc/data/papers/SP/moe_transformer_dag', format='png', cleanup=False)

print("MoE Transformer DAG generated successfully!")
print("Files created:")
print("- moe_transformer_dag.dot")
print("- moe_transformer_dag.png")