import graphviz

# Create MA Separation DAG
dot = graphviz.Digraph('MA_Separation', comment='MA Separation Parallel Strategy for MoE-Attention Co-execution')
dot.attr(rankdir='TB', size='20,20')

# Define node shapes
dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
dot.attr('node', shape='diamond', style='filled', fillcolor='lightcoral')  # Communication

# Input layer
dot.node('input', 'Model Input\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: all', shape='ellipse')

# Layer 1 - Attention Phase (GPUs 0-7)
for gpu_id in range(8):
    # QKV Projection
    dot.node(f'l1_qkv_proj_{gpu_id}', f'L1 QKV Projection\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, heads=4, d_k=128]\\nGPU: {gpu_id}', shape='rectangle')
    
    # Attention Score Computation
    dot.node(f'l1_attn_score_{gpu_id}', f'L1 Attention Score\\nInput: [batch_size=1, seq_len=2048, heads=4, d_k=128]\\nOutput: [batch_size=1, seq_len=2048, heads=4, d_k=128]\\nGPU: {gpu_id}', shape='rectangle')
    
    # Attention Output
    dot.node(f'l1_attn_out_{gpu_id}', f'L1 Attention Output\\nInput: [batch_size=1, seq_len=2048, heads=4, d_k=128]\\nOutput: [batch_size=1, seq_len=2048, hidden=512]\\nGPU: {gpu_id}', shape='rectangle')

# Layer 1 - Attention Aggregation
for gpu_id in range(8):
    dot.node(f'l1_attn_agg_{gpu_id}', f'L1 Attention Aggregate\\nInput: [batch_size=1, seq_len=2048, hidden=512]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='parallelogram')

# Layer 1 - Residual Connection
for gpu_id in range(8):
    dot.node(f'l1_residual_{gpu_id}', f'L1 Residual Add\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')

# Layer 1 - MoE Phase (GPUs 8-15)
for gpu_id in range(8, 16):
    # Gating Network
    dot.node(f'l1_gate_{gpu_id}', f'L1 Gating Network\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, experts=2]\\nGPU: {gpu_id}', shape='parallelogram')
    
    # Expert 1
    dot.node(f'l1_expert1_{gpu_id}', f'L1 Expert 1\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')
    
    # Expert 2  
    dot.node(f'l1_expert2_{gpu_id}', f'L1 Expert 2\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')
    
    # Expert Aggregation
    dot.node(f'l1_expert_agg_{gpu_id}', f'L1 Expert Aggregate\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='parallelogram')

# Layer 1 - MoE Residual
for gpu_id in range(8, 16):
    dot.node(f'l1_moe_residual_{gpu_id}', f'L1 MoE Residual Add\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')

# Repeat for Layers 2-4 (similar structure)
for layer in range(2, 5):
    # Attention Phase (GPUs 0-7)
    for gpu_id in range(8):
        # QKV Projection
        dot.node(f'l{layer}_qkv_proj_{gpu_id}', f'L{layer} QKV Projection\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, heads=4, d_k=128]\\nGPU: {gpu_id}', shape='rectangle')
        
        # Attention Score Computation
        dot.node(f'l{layer}_attn_score_{gpu_id}', f'L{layer} Attention Score\\nInput: [batch_size=1, seq_len=2048, heads=4, d_k=128]\\nOutput: [batch_size=1, seq_len=2048, heads=4, d_k=128]\\nGPU: {gpu_id}', shape='rectangle')
        
        # Attention Output
        dot.node(f'l{layer}_attn_out_{gpu_id}', f'L{layer} Attention Output\\nInput: [batch_size=1, seq_len=2048, heads=4, d_k=128]\\nOutput: [batch_size=1, seq_len=2048, hidden=512]\\nGPU: {gpu_id}', shape='rectangle')

    # Layer Attention Aggregation
    for gpu_id in range(8):
        dot.node(f'l{layer}_attn_agg_{gpu_id}', f'L{layer} Attention Aggregate\\nInput: [batch_size=1, seq_len=2048, hidden=512]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='parallelogram')

    # Layer Residual Connection
    for gpu_id in range(8):
        dot.node(f'l{layer}_residual_{gpu_id}', f'L{layer} Residual Add\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')

    # MoE Phase (GPUs 8-15)
    for gpu_id in range(8, 16):
        # Gating Network
        dot.node(f'l{layer}_gate_{gpu_id}', f'L{layer} Gating Network\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, experts=2]\\nGPU: {gpu_id}', shape='parallelogram')
        
        # Expert 1
        dot.node(f'l{layer}_expert1_{gpu_id}', f'L{layer} Expert 1\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')
        
        # Expert 2  
        dot.node(f'l{layer}_expert2_{gpu_id}', f'L{layer} Expert 2\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')
        
        # Expert Aggregation
        dot.node(f'l{layer}_expert_agg_{gpu_id}', f'L{layer} Expert Aggregate\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='parallelogram')

    # MoE Residual
    for gpu_id in range(8, 16):
        dot.node(f'l{layer}_moe_residual_{gpu_id}', f'L{layer} MoE Residual Add\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPU: {gpu_id}', shape='rectangle')

# Output layer
dot.node('output', 'Model Output\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, vocab_size]\\nGPU: 15', shape='ellipse')

# Communication nodes for attention all-reduce
for layer in range(1, 5):
    for gpu_id in range(8):
        dot.node(f'l{layer}_attn_comm_{gpu_id}', f'L{layer} Attention All-Reduce\\nInput: [batch_size=1, seq_len=2048, hidden=512]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPUs: 0-7', shape='diamond')

# Communication nodes for MoE all-to-all
for layer in range(1, 5):
    for gpu_id in range(8, 16):
        dot.node(f'l{layer}_moe_comm_{gpu_id}', f'L{layer} MoE All-to-All\\nInput: [batch_size=1, seq_len=2048, hidden=4096]\\nOutput: [batch_size=1, seq_len=2048, hidden=4096]\\nGPUs: 8-15', shape='diamond')

# Connect the DAG
# Input to Layer 1 Attention
dot.edge('input', 'l1_qkv_proj_0')
dot.edge('input', 'l1_qkv_proj_1')
dot.edge('input', 'l1_qkv_proj_2')
dot.edge('input', 'l1_qkv_proj_3')
dot.edge('input', 'l1_qkv_proj_4')
dot.edge('input', 'l1_qkv_proj_5')
dot.edge('input', 'l1_qkv_proj_6')
dot.edge('input', 'l1_qkv_proj_7')

# Layer 1 Attention connections
for gpu_id in range(8):
    dot.edge(f'l1_qkv_proj_{gpu_id}', f'l1_attn_score_{gpu_id}')
    dot.edge(f'l1_attn_score_{gpu_id}', f'l1_attn_out_{gpu_id}')
    dot.edge(f'l1_attn_out_{gpu_id}', f'l1_attn_comm_{gpu_id}')
    dot.edge(f'l1_attn_comm_{gpu_id}', f'l1_attn_agg_{gpu_id}')
    dot.edge(f'l1_attn_agg_{gpu_id}', f'l1_residual_{gpu_id}')
    dot.edge('input', f'l1_residual_{gpu_id}')  # Residual connection

# Layer 1 to MoE transition
for gpu_id in range(8):
    for moe_gpu in range(8, 16):
        dot.edge(f'l1_residual_{gpu_id}', f'l1_gate_{moe_gpu}')

# Layer 1 MoE connections
for gpu_id in range(8, 16):
    dot.edge(f'l1_gate_{gpu_id}', f'l1_expert1_{gpu_id}', style='dashed')
    dot.edge(f'l1_gate_{gpu_id}', f'l1_expert2_{gpu_id}', style='dashed')
    dot.edge(f'l1_expert1_{gpu_id}', f'l1_expert_agg_{gpu_id}')
    dot.edge(f'l1_expert2_{gpu_id}', f'l1_expert_agg_{gpu_id}')
    dot.edge(f'l1_expert_agg_{gpu_id}', f'l1_moe_comm_{gpu_id}')
    dot.edge(f'l1_moe_comm_{gpu_id}', f'l1_moe_residual_{gpu_id}')
    # Connect from attention GPUs to MoE GPUs for residual
    for attn_gpu in range(8):
        dot.edge(f'l1_residual_{attn_gpu}', f'l1_moe_residual_{gpu_id}')

# Connect layers 2-4 (similar pattern)
for layer in range(2, 5):
    prev_layer = layer - 1
    
    # Connect MoE output to next layer attention
    for moe_gpu in range(8, 16):
        for attn_gpu in range(8):
            dot.edge(f'l{prev_layer}_moe_residual_{moe_gpu}', f'l{layer}_qkv_proj_{attn_gpu}')
    
    # Attention connections
    for gpu_id in range(8):
        dot.edge(f'l{layer}_qkv_proj_{gpu_id}', f'l{layer}_attn_score_{gpu_id}')
        dot.edge(f'l{layer}_attn_score_{gpu_id}', f'l{layer}_attn_out_{gpu_id}')
        dot.edge(f'l{layer}_attn_out_{gpu_id}', f'l{layer}_attn_comm_{gpu_id}')
        dot.edge(f'l{layer}_attn_comm_{gpu_id}', f'l{layer}_attn_agg_{gpu_id}')
        dot.edge(f'l{layer}_attn_agg_{gpu_id}', f'l{layer}_residual_{gpu_id}')
        # Connect from previous layer to residual
        for prev_moe_gpu in range(8, 16):
            dot.edge(f'l{prev_layer}_moe_residual_{prev_moe_gpu}', f'l{layer}_residual_{gpu_id}')
    
    # MoE connections
    for gpu_id in range(8, 16):
        dot.edge(f'l{layer}_gate_{gpu_id}', f'l{layer}_expert1_{gpu_id}', style='dashed')
        dot.edge(f'l{layer}_gate_{gpu_id}', f'l{layer}_expert2_{gpu_id}', style='dashed')
        dot.edge(f'l{layer}_expert1_{gpu_id}', f'l{layer}_expert_agg_{gpu_id}')
        dot.edge(f'l{layer}_expert2_{gpu_id}', f'l{layer}_expert_agg_{gpu_id}')
        dot.edge(f'l{layer}_expert_agg_{gpu_id}', f'l{layer}_moe_comm_{gpu_id}')
        dot.edge(f'l{layer}_moe_comm_{gpu_id}', f'l{layer}_moe_residual_{gpu_id}')
        # Connect from attention GPUs to MoE GPUs
        for attn_gpu in range(8):
            dot.edge(f'l{layer}_residual_{attn_gpu}', f'l{layer}_moe_residual_{gpu_id}')

# Connect final layer to output
for gpu_id in range(8, 16):
    dot.edge(f'l4_moe_residual_{gpu_id}', 'output')

# Save the DAG
dot.render('/home/wzc/data/file-share/2025-09-30-10-14-27/ma_separation', format='svg', cleanup=False)
dot.save('/home/wzc/data/file-share/2025-09-30-10-14-27/ma_separation.dot')

print("MA Separation DAG generated successfully!")