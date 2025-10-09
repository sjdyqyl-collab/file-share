import graphviz

def create_fa_pool_dag():
    dot = graphviz.Digraph('fa_pool_model', comment='FA Pool Dynamic Parallel Strategy DAG')
    
    # Set graph attributes
    dot.attr(rankdir='TB', size='25,35')
    dot.attr('node', shape='rectangle', style='filled')
    
    # Input layer
    dot.node('input', 'Input\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: All GPUs', shape='ellipse', fillcolor='lightblue')
    
    # Sequence length monitor
    dot.node('monitor', 'Sequence Length Monitor\nInput: [batch_size=B, seq_len=S]\nOutput: [threshold_check=bool]\nGPU: 0', shape='diamond', fillcolor='gold')
    
    # Base layer - 8 GPUs (always active)
    dot.node('embed', 'Embedding (Base)\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', fillcolor='lightgreen')
    dot.node('ln1_base', 'LayerNorm (Base)\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', fillcolor='lightyellow')
    
    # FFN layers (base - always on 8 GPUs)
    dot.node('ffn1', 'FFN Layer 1\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', fillcolor='lightpink')
    dot.node('ffn2', 'FFN Layer 2\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', fillcolor='lightpink')
    dot.node('ffn3', 'FFN Layer 3\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', fillcolor='lightpink')
    dot.node('ffn4', 'FFN Layer 4\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', fillcolor='lightpink')
    
    # Attention Pool - Dynamic allocation
    dot.node('pool_gate', 'Pool Gate\nInput: [threshold_check=bool]\nOutput: [pool_activation=bool]\nGPU: 0', shape='parallelogram', fillcolor='gold')
    
    # When sequence <= 4096: Use base 8 GPUs
    dot.node('q_proj_base', 'Q Projection (Base)\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nGPU: 0-7 (TP=8)', fillcolor='lightcoral')
    dot.node('k_proj_base', 'K Projection (Base)\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nGPU: 0-7 (TP=8)', fillcolor='lightcoral')
    dot.node('v_proj_base', 'V Projection (Base)\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nGPU: 0-7 (TP=8)', fillcolor='lightcoral')
    dot.node('flash_attn_base', 'Flash Attention (Base)\nInput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nOutput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nGPU: 0-7 (TP=8)', fillcolor='orange')
    dot.node('o_proj_base', 'O Projection (Base)\nInput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7 (TP=8)', fillcolor='lightcoral')
    
    # When sequence > 4096: Use attention pool (32 GPUs)
    dot.node('split_tokens', 'Token Splitter\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S/p, hidden_size=4096]\nGPU: 8-39', shape='parallelogram', fillcolor='gold')
    
    # Attention pool projections (distributed across 32 GPUs)
    for i in range(32):
        gpu_id = 8 + i
        dot.node(f'q_proj_pool_{i}', f'Q Projection Pool GPU {i}\nInput: [batch_size=B, seq_len=S/32, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S/32, heads=32, d_k=128]\nGPU: {gpu_id}', fillcolor='lightcoral')
        dot.node(f'k_proj_pool_{i}', f'K Projection Pool GPU {i}\nInput: [batch_size=B, seq_len=S/32, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S/32, heads=32, d_k=128]\nGPU: {gpu_id}', fillcolor='lightcoral')
        dot.node(f'v_proj_pool_{i}', f'V Projection Pool GPU {i}\nInput: [batch_size=B, seq_len=S/32, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S/32, heads=32, d_k=128]\nGPU: {gpu_id}', fillcolor='lightcoral')
        dot.node(f'flash_attn_pool_{i}', f'Flash Attention Pool GPU {i}\nInput: [batch_size=B, seq_len=S/32, heads=32, d_k=128]\nOutput: [batch_size=B, seq_len=S/32, heads=32, d_k=128]\nGPU: {gpu_id}', fillcolor='orange')
        dot.node(f'o_proj_pool_{i}', f'O Projection Pool GPU {i}\nInput: [batch_size=B, seq_len=S/32, heads=32, d_k=128]\nOutput: [batch_size=B, seq_len=S/32, hidden_size=4096]\nGPU: {gpu_id}', fillcolor='lightcoral')
    
    # Attention pool aggregation
    dot.node('concat_attn', 'Attention Concat\nInput: [batch_size=B, seq_len=S/32, hidden_size=4096]×32\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', shape='parallelogram', fillcolor='lightsteelblue')
    
    # KV cache sharing
    dot.node('kv_cache', 'KV Cache Share\nInput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nOutput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nGPU: 8-39', shape='parallelogram', style='dashed', fillcolor='lightsteelblue')
    
    # Residual connections and layer norm
    dot.node('res1', 'Residual Add Layer 1\nInput: [batch_size=B, seq_len=S, hidden_size=4096], [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', fillcolor='lightgray')
    dot.node('ln1', 'LayerNorm Layer 1\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', fillcolor='lightyellow')
    
    dot.node('res2', 'Residual Add Layer 2\nInput: [batch_size=B, seq_len=S, hidden_size=4096], [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', fillcolor='lightgray')
    dot.node('ln2', 'LayerNorm Layer 2\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', fillcolor='lightyellow')
    
    dot.node('res3', 'Residual Add Layer 3\nInput: [batch_size=B, seq_len=S, hidden_size=4096], [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', fillcolor='lightgray')
    dot.node('ln3', 'LayerNorm Layer 3\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', fillcolor='lightyellow')
    
    dot.node('res4', 'Residual Add Layer 4\nInput: [batch_size=B, seq_len=S, hidden_size=4096], [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', fillcolor='lightgray')
    
    # Output
    dot.node('output', 'Output\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, vocab_size=V]\nGPU: 0-7', shape='ellipse', fillcolor='lightblue')
    
    # Resource manager
    dot.node('resource_mgr', 'Resource Manager\nInput: [pool_activation=bool]\nOutput: [gpu_allocation=32]\nGPU: 0', shape='hexagon', fillcolor='gold')
    
    # Create edges for base flow
    dot.edge('input', 'monitor')
    dot.edge('monitor', 'pool_gate')
    dot.edge('input', 'embed')
    dot.edge('embed', 'ln1_base')
    
    # Base attention path (sequence <= 4096)
    dot.edge('pool_gate', 'q_proj_base', label='seq <= 4096')
    dot.edge('pool_gate', 'k_proj_base', label='seq <= 4096')
    dot.edge('pool_gate', 'v_proj_base', label='seq <= 4096')
    dot.edge('ln1_base', 'q_proj_base')
    dot.edge('ln1_base', 'k_proj_base')
    dot.edge('ln1_base', 'v_proj_base')
    dot.edge('q_proj_base', 'flash_attn_base')
    dot.edge('k_proj_base', 'flash_attn_base')
    dot.edge('v_proj_base', 'flash_attn_base')
    dot.edge('flash_attn_base', 'o_proj_base')
    dot.edge('o_proj_base', 'res1')
    dot.edge('ln1_base', 'res1')  # Residual
    
    # Pool attention path (sequence > 4096)
    dot.edge('pool_gate', 'resource_mgr', label='seq > 4096')
    dot.edge('resource_mgr', 'split_tokens')
    dot.edge('ln1_base', 'split_tokens')
    
    # Connect each pool GPU
    for i in range(32):
        dot.edge('split_tokens', f'q_proj_pool_{i}')
        dot.edge('split_tokens', f'k_proj_pool_{i}')
        dot.edge('split_tokens', f'v_proj_pool_{i}')
        dot.edge(f'q_proj_pool_{i}', f'flash_attn_pool_{i}')
        dot.edge(f'k_proj_pool_{i}', f'flash_attn_pool_{i}')
        dot.edge(f'v_proj_pool_{i}', f'flash_attn_pool_{i}')
        dot.edge(f'flash_attn_pool_{i}', f'o_proj_pool_{i}')
        dot.edge(f'o_proj_pool_{i}', 'concat_attn')
        
        # KV cache sharing (dashed)
        dot.edge('kv_cache', f'flash_attn_pool_{i}', style='dashed')
    
    dot.edge('concat_attn', 'res1')
    dot.edge('ln1_base', 'res1')  # Residual
    
    # Continue with FFN layers
    dot.edge('res1', 'ffn1')
    dot.edge('ffn1', 'res2')
    dot.edge('res1', 'res2')  # Residual
    dot.edge('res2', 'ln1')
    
    # Layer 2
    dot.edge('ln1', 'ffn2')
    dot.edge('ffn2', 'res3')
    dot.edge('ln1', 'res3')  # Residual
    dot.edge('res3', 'ln2')
    
    # Layer 3
    dot.edge('ln2', 'ffn3')
    dot.edge('ffn3', 'res4')
    dot.edge('ln2', 'res4')  # Residual
    dot.edge('res4', 'ln3')
    
    # Layer 4
    dot.edge('ln3', 'ffn4')
    dot.edge('ffn4', 'output')
    
    return dot

if __name__ == "__main__":
    dag = create_fa_pool_dag()
    dag.render('/home/wzc/data/file-share/logs/2025-10-09-15-07-03/fa_pool_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/logs/2025-10-09-15-07-03/fa_pool_dag.dot')
    print("FA Pool DAG generated successfully")