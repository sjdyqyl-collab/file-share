import graphviz

def create_dense_model_dag():
    """
    Generate DAG for 16-layer dense model with layer-wise partitioning across 16 GPUs
    Each GPU gets exactly one layer
    """
    dot = graphviz.Digraph(comment='Dense Model Layer-wise Deployment', format='svg')
    dot.attr(rankdir='TB', size='20,20')
    
    # Model parameters
    batch_size = 1024
    seq_len = 2048
    hidden_size = 8192  # 16 heads * 512 per head
    ffn_hidden_size = 32768
    num_layers = 16
    num_gpus = 16
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    dot.attr('edge', fontsize='10')
    
    # Input node
    dot.node('input', f'Total Input\\nShape: [{batch_size}, {seq_len}, {hidden_size}]\\nGPU: Host', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Create nodes for each layer on each GPU
    for layer_idx in range(num_layers):
        gpu_id = layer_idx  # Each layer on separate GPU
        
        # Layer input (from previous layer or input)
        if layer_idx == 0:
            layer_input = 'input'
        else:
            layer_input = f'comm_{layer_idx-1}_{layer_idx}'
        
        # Communication node between layers
        if layer_idx > 0:
            dot.node(f'comm_{layer_idx-1}_{layer_idx}', 
                     f'Layer Transfer\\nShape: [{batch_size}, {seq_len}, {hidden_size}]\\nGPU {layer_idx-1} → GPU {layer_idx}',
                     shape='parallelogram', fillcolor='yellow')
            dot.edge(f'layer_{layer_idx-1}_add', f'comm_{layer_idx-1}_{layer_idx}')
        
        # Layer normalization 1
        dot.node(f'layer_{layer_idx}_ln1', 
                 f'LayerNorm 1\\nShape: [{batch_size}, {seq_len}, {hidden_size}]\\nGPU {gpu_id}',
                 fillcolor='lightcoral')
        dot.edge(layer_input, f'layer_{layer_idx}_ln1')
        
        # Multi-head attention
        # Q projection
        dot.node(f'layer_{layer_idx}_q_proj', 
                 f'Q Projection\\nShape: [{batch_size}, {seq_len}, {hidden_size}]\\nGPU {gpu_id}',
                 fillcolor='lightblue')
        dot.edge(f'layer_{layer_idx}_ln1', f'layer_{layer_idx}_q_proj')
        
        # K projection
        dot.node(f'layer_{layer_idx}_k_proj', 
                 f'K Projection\\nShape: [{batch_size}, {seq_len}, {hidden_size}]\\nGPU {gpu_id}',
                 fillcolor='lightblue')
        dot.edge(f'layer_{layer_idx}_ln1', f'layer_{layer_idx}_k_proj')
        
        # V projection
        dot.node(f'layer_{layer_idx}_v_proj', 
                 f'V Projection\\nShape: [{batch_size}, {seq_len}, {hidden_size}]\\nGPU {gpu_id}',
                 fillcolor='lightblue')
        dot.edge(f'layer_{layer_idx}_ln1', f'layer_{layer_idx}_v_proj')
        
        # Reshape for multi-head
        dot.node(f'layer_{layer_idx}_q_reshape', 
                 f'Reshape Q\\nShape: [{batch_size}, 16, {seq_len}, 512]\\nGPU {gpu_id}',
                 fillcolor='lightblue')
        dot.edge(f'layer_{layer_idx}_q_proj', f'layer_{layer_idx}_q_reshape')
        
        dot.node(f'layer_{layer_idx}_k_reshape', 
                 f'Reshape K\\nShape: [{batch_size}, 16, {seq_len}, 512]\\nGPU {gpu_id}',
                 fillcolor='lightblue')
        dot.edge(f'layer_{layer_idx}_k_proj', f'layer_{layer_idx}_k_reshape')
        
        dot.node(f'layer_{layer_idx}_v_reshape', 
                 f'Reshape V\\nShape: [{batch_size}, 16, {seq_len}, 512]\\nGPU {gpu_id}',
                 fillcolor='lightblue')
        dot.edge(f'layer_{layer_idx}_v_proj', f'layer_{layer_idx}_v_reshape')
        
        # Attention computation
        dot.node(f'layer_{layer_idx}_attn_scores', 
                 f'QK^T Matmul\\nShape: [{batch_size}, 16, {seq_len}, {seq_len}]\\nGPU {gpu_id}',
                 fillcolor='lightblue')
        dot.edge(f'layer_{layer_idx}_q_reshape', f'layer_{layer_idx}_attn_scores')
        dot.edge(f'layer_{layer_idx}_k_reshape', f'layer_{layer_idx}_attn_scores')
        
        dot.node(f'layer_{layer_idx}_attn_weights', 
                 f'Softmax\\nShape: [{batch_size}, 16, {seq_len}, {seq_len}]\\nGPU {gpu_id}',
                 fillcolor='lightblue')
        dot.edge(f'layer_{layer_idx}_attn_scores', f'layer_{layer_idx}_attn_weights')
        
        dot.node(f'layer_{layer_idx}_attn_output', 
                 f'Weighted V\\nShape: [{batch_size}, 16, {seq_len}, 512]\\nGPU {gpu_id}',
                 fillcolor='lightblue')
        dot.edge(f'layer_{layer_idx}_attn_weights', f'layer_{layer_idx}_attn_output')
        dot.edge(f'layer_{layer_idx}_v_reshape', f'layer_{layer_idx}_attn_output')
        
        # Reshape back
        dot.node(f'layer_{layer_idx}_attn_reshape', 
                 f'Reshape Back\\nShape: [{batch_size}, {seq_len}, {hidden_size}]\\nGPU {gpu_id}',
                 fillcolor='lightblue')
        dot.edge(f'layer_{layer_idx}_attn_output', f'layer_{layer_idx}_attn_reshape')
        
        # Output projection
        dot.node(f'layer_{layer_idx}_attn_proj', 
                 f'Output Projection\\nShape: [{batch_size}, {seq_len}, {hidden_size}]\\nGPU {gpu_id}',
                 fillcolor='lightblue')
        dot.edge(f'layer_{layer_idx}_attn_reshape', f'layer_{layer_idx}_attn_proj')
        
        # Residual connection 1
        dot.node(f'layer_{layer_idx}_add1', 
                 f'Residual Add 1\\nShape: [{batch_size}, {seq_len}, {hidden_size}]\\nGPU {gpu_id}',
                 fillcolor='lightgreen')
        dot.edge(layer_input, f'layer_{layer_idx}_add1')
        dot.edge(f'layer_{layer_idx}_attn_proj', f'layer_{layer_idx}_add1')
        
        # Layer normalization 2
        dot.node(f'layer_{layer_idx}_ln2', 
                 f'LayerNorm 2\\nShape: [{batch_size}, {seq_len}, {hidden_size}]\\nGPU {gpu_id}',
                 fillcolor='lightcoral')
        dot.edge(f'layer_{layer_idx}_add1', f'layer_{layer_idx}_ln2')
        
        # FFN - First linear (column parallel)
        dot.node(f'layer_{layer_idx}_ffn1', 
                 f'FFN First Linear\\nShape: [{batch_size}, {seq_len}, {ffn_hidden_size}]\\nGPU {gpu_id}',
                 fillcolor='lightblue')
        dot.edge(f'layer_{layer_idx}_ln2', f'layer_{layer_idx}_ffn1')
        
        # GELU activation
        dot.node(f'layer_{layer_idx}_gelu', 
                 f'GELU Activation\\nShape: [{batch_size}, {seq_len}, {ffn_hidden_size}]\\nGPU {gpu_id}',
                 fillcolor='lightblue')
        dot.edge(f'layer_{layer_idx}_ffn1', f'layer_{layer_idx}_gelu')
        
        # FFN - Second linear (row parallel)
        dot.node(f'layer_{layer_idx}_ffn2', 
                 f'FFN Second Linear\\nShape: [{batch_size}, {seq_len}, {hidden_size}]\\nGPU {gpu_id}',
                 fillcolor='lightblue')
        dot.edge(f'layer_{layer_idx}_gelu', f'layer_{layer_idx}_ffn2')
        
        # Residual connection 2
        dot.node(f'layer_{layer_idx}_add', 
                 f'Residual Add 2\\nShape: [{batch_size}, {seq_len}, {hidden_size}]\\nGPU {gpu_id}',
                 fillcolor='lightgreen')
        dot.edge(f'layer_{layer_idx}_add1', f'layer_{layer_idx}_add')
        dot.edge(f'layer_{layer_idx}_ffn2', f'layer_{layer_idx}_add')
    
    # Output node
    dot.node('output', f'Total Output\\nShape: [{batch_size}, {seq_len}, {hidden_size}]\\nGPU: Host', 
             shape='ellipse', fillcolor='lightgreen')
    dot.edge(f'comm_{num_layers-1}_{num_layers}', 'output')
    
    # Save the DAG
    dot.render('/home/wzc/data/file-share/submission/dense_model_dag', cleanup=True)
    return '/home/wzc/data/file-share/submission/dense_model_dag.svg'

if __name__ == '__main__':
    create_dense_model_dag()