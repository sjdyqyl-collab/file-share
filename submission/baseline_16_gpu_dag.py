import graphviz

# Create the DAG for the baseline deployment (16 GPUs, TP=8, PP=2, 4 experts per GPU)
dot = graphviz.Digraph(comment='Baseline Deployment - 16 GPUs (TP=8, PP=2)')
dot.attr(rankdir='TB', size='30,20')

# Define node styles
dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication nodes
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation nodes
dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/aggregation nodes

# Model parameters
num_layers = 4
experts_per_layer = 16
experts_per_gpu = 4
total_gpus = 16
tp_degree = 8
pp_degree = 2
batch_size = 1024
seq_len = 1
hidden_size = 8192
mlp_hidden_size = 32768

# Calculate pipeline stages
layers_per_stage = num_layers // pp_degree

# Input processing
with dot.subgraph(name='cluster_input') as c:
    c.attr(label='Input Processing', style='dashed')
    c.node('input_tokens', f'Input Tokens\\nShape: ({batch_size}, {seq_len})\\nGPU: CPU', shape='ellipse', fillcolor='lightblue')
    c.node('embed', f'Embedding Layer\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: Stage 0', shape='rectangle', fillcolor='lightgreen')
    c.node('pos_embed', f'Positional Encoding\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: Stage 0', shape='rectangle', fillcolor='lightgreen')
    
    dot.edge('input_tokens', 'embed')
    dot.edge('embed', 'pos_embed')

prev_output = 'pos_embed'

# Process pipeline stages
for stage_idx in range(pp_degree):
    stage_start_layer = stage_idx * layers_per_stage
    stage_end_layer = (stage_idx + 1) * layers_per_stage
    
    with dot.subgraph(name=f'cluster_stage_{stage_idx}') as c:
        c.attr(label=f'Pipeline Stage {stage_idx}\\n(GPUs {stage_idx * 8}-{(stage_idx + 1) * 8 - 1})', style='dashed')
        
        # Process each layer in this stage
        for layer_idx in range(stage_start_layer, stage_end_layer):
            with dot.subgraph(name=f'cluster_stage_{stage_idx}_layer_{layer_idx}') as lc:
                lc.attr(label=f'Layer {layer_idx + 1}', style='dotted')
                
                # MHA computation (tensor parallel across 8 GPUs)
                mha_qkv = f'mha_qkv_{layer_idx}'
                mha_attn = f'mha_attn_{layer_idx}'
                mha_out = f'mha_out_{layer_idx}'
                mha_residual = f'mha_residual_{layer_idx}'
                
                lc.node(mha_qkv, f'MHA QKV Linear (TP=8)\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {stage_idx * 8}-{(stage_idx + 1) * 8 - 1}', 
                       shape='rectangle', fillcolor='lightgreen')
                lc.node(mha_attn, f'MHA Attention (TP=8)\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {stage_idx * 8}-{(stage_idx + 1) * 8 - 1}', 
                       shape='rectangle', fillcolor='lightgreen')
                lc.node(mha_out, f'MHA Output Linear (TP=8)\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {stage_idx * 8}-{(stage_idx + 1) * 8 - 1}', 
                        shape='rectangle', fillcolor='lightgreen')
                lc.node(mha_residual, f'MHA Residual Add\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {stage_idx * 8}-{(stage_idx + 1) * 8 - 1}', 
                         shape='parallelogram', fillcolor='lightyellow')
                
                dot.edge(prev_output, mha_qkv)
                dot.edge(mha_qkv, mha_attn)
                dot.edge(mha_attn, mha_out)
                dot.edge(mha_out, mha_residual)
                dot.edge(prev_output, mha_residual)  # Residual connection
                
                # Layer normalization after MHA
                ln1 = f'ln1_{layer_idx}'
                lc.node(ln1, f'Layer Norm 1\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {stage_idx * 8}-{(stage_idx + 1) * 8 - 1}', 
                         shape='rectangle', fillcolor='lightgreen')
                dot.edge(mha_residual, ln1)
                
                # MoE layer - 4 experts per GPU
                gate = f'gate_{layer_idx}'
                lc.node(gate, f'Expert Gate\\nShape: ({batch_size}, {seq_len}, {experts_per_layer})\\nGPU: {stage_idx * 8}-{(stage_idx + 1) * 8 - 1}', 
                         shape='parallelogram', fillcolor='lightyellow')
                dot.edge(ln1, gate)
                
                # Expert computation (4 experts per GPU, 16 total experts)
                expert_outputs = []
                for gpu_idx in range(stage_idx * 8, (stage_idx + 1) * 8):
                    for expert_idx_in_gpu in range(experts_per_gpu):
                        expert_global_idx = (gpu_idx - stage_idx * 8) * experts_per_gpu + expert_idx_in_gpu
                        
                        # Token routing to this expert
                        route_to_expert = f'route_to_expert_{layer_idx}_{gpu_idx}_{expert_idx_in_gpu}'
                        lc.node(route_to_expert, 
                                 f'Route to Expert {expert_global_idx}\\nShape: (tokens, {hidden_size})\\nGPU: {gpu_idx}', 
                                 shape='ellipse', fillcolor='lightblue')
                        dot.edge(gate, route_to_expert, style='dashed')
                        
                        # Expert computation on this GPU
                        expert_linear1 = f'expert_linear1_{layer_idx}_{gpu_idx}_{expert_idx_in_gpu}'
                        expert_activation = f'expert_activation_{layer_idx}_{gpu_idx}_{expert_idx_in_gpu}'
                        expert_linear2 = f'expert_linear2_{layer_idx}_{gpu_idx}_{expert_idx_in_gpu}'
                        
                        lc.node(expert_linear1, 
                                f'Expert {expert_global_idx} Linear 1\\nShape: (tokens, {mlp_hidden_size})\\nGPU: {gpu_idx}', 
                                shape='rectangle', fillcolor='lightgreen')
                        lc.node(expert_activation, 
                                f'Expert {expert_global_idx} GELU\\nShape: (tokens, {mlp_hidden_size})\\nGPU: {gpu_idx}', 
                                shape='rectangle', fillcolor='lightgreen')
                        lc.node(expert_linear2, 
                                f'Expert {expert_global_idx} Linear 2\\nShape: (tokens, {hidden_size})\\nGPU: {gpu_idx}', 
                                shape='rectangle', fillcolor='lightgreen')
                        
                        dot.edge(route_to_expert, expert_linear1)
                        dot.edge(expert_linear1, expert_activation)
                        dot.edge(expert_activation, expert_linear2)
                        
                        expert_outputs.append(expert_linear2)
                
                # Aggregate expert outputs (within each tensor parallel group)
                aggregate = f'aggregate_{layer_idx}'
                lc.node(aggregate, 
                        f'Aggregate Expert Outputs\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {stage_idx * 8}-{(stage_idx + 1) * 8 - 1}', 
                        shape='parallelogram', fillcolor='lightyellow')
                
                for expert_out in expert_outputs:
                    dot.edge(expert_out, aggregate)
                
                # Final residual connection and layer norm
                moe_residual = f'moe_residual_{layer_idx}'
                ln2 = f'ln2_{layer_idx}'
                
                lc.node(moe_residual, 
                        f'MoE Residual Add\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {stage_idx * 8}-{(stage_idx + 1) * 8 - 1}', 
                        shape='parallelogram', fillcolor='lightyellow')
                lc.node(ln2, 
                        f'Layer Norm 2\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {stage_idx * 8}-{(stage_idx + 1) * 8 - 1}', 
                        shape='rectangle', fillcolor='lightgreen')
                
                dot.edge(ln1, moe_residual)
                dot.edge(aggregate, moe_residual)
                dot.edge(moe_residual, ln2)
                
                prev_output = ln2
        
        # Pipeline communication between stages
        if stage_idx < pp_degree - 1:
            pipeline_comm = f'pipeline_comm_{stage_idx}'
            c.node(pipeline_comm, 
                   f'Pipeline Communication\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: Stage {stage_idx} -> Stage {stage_idx + 1}', 
                   shape='ellipse', fillcolor='lightblue')
            dot.edge(prev_output, pipeline_comm)
            prev_output = pipeline_comm

# Output processing (on last stage)
with dot.subgraph(name='cluster_output') as c:
    c.attr(label='Output Processing', style='dashed')
    
    final_norm = 'final_norm'
    output_linear = 'output_linear'
    softmax = 'softmax'
    
    c.node(final_norm, f'Final Layer Norm\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: Stage 1', shape='rectangle', fillcolor='lightgreen')
    c.node(output_linear, f'Output Linear\\nShape: ({batch_size}, {seq_len}, vocab_size)\\nGPU: Stage 1', shape='rectangle', fillcolor='lightgreen')
    c.node(softmax, f'Softmax\\nShape: ({batch_size}, {seq_len}, vocab_size)\\nGPU: Stage 1', shape='rectangle', fillcolor='lightgreen')
    
    dot.edge(prev_output, final_norm)
    dot.edge(final_norm, output_linear)
    dot.edge(output_linear, softmax)

# Save the DAG
dot.render('/home/wzc/data/file-share/submission/baseline_16_gpu_dag', format='svg', cleanup=False)
print("Generated: baseline_16_gpu_dag.svg")