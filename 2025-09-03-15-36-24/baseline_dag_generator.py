#!/usr/bin/env python3
import graphviz

# Create baseline DAG with 16 GPUs (TP=8, PP=2)
# Each GPU has 4 experts

dot = graphviz.Digraph('baseline_moe_dag', comment='Baseline MoE with TP=8, PP=2, 16 GPUs')
dot.attr(rankdir='TB', size='20,30')

# Define node styles
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
dot.attr('edge', fontsize='10')

# Input node
dot.node('input', 'Input\n[1024, hidden_size]\nAll GPUs', shape='ellipse', fillcolor='lightgreen')

# Pipeline stages
for stage in [0, 1]:
    with dot.subgraph(name=f'cluster_stage_{stage}') as c:
        c.attr(label=f'Pipeline Stage {stage}', style='dashed')
        
        # Each stage has 2 layers
        for layer in range(2):
            layer_id = stage * 2 + layer
            
            with c.subgraph(name=f'cluster_layer_{layer_id}') as layer_cluster:
                layer_cluster.attr(label=f'Layer {layer_id}', style='rounded')
                
                # MHA across 8 GPUs with tensor parallelism
                with layer_cluster.subgraph(name=f'cluster_mha_{layer_id}') as mha_cluster:
                    mha_cluster.attr(label='Multi-Head Attention (TP=8)', style='dotted')
                    
                    # QKV projection (column parallel)
                    for gpu in range(8):
                        gpu_id = stage * 8 + gpu
                        mha_cluster.node(f'qkv_{layer_id}_{gpu}', 
                                       f'QKV Proj\n[1024, 768]\nGPU {gpu_id}',
                                       fillcolor='lightyellow')
                    
                    # Attention computation
                    for gpu in range(8):
                        gpu_id = stage * 8 + gpu
                        mha_cluster.node(f'attn_{layer_id}_{gpu}', 
                                       f'Attention\n[1024, 512]\nGPU {gpu_id}',
                                       fillcolor='lightcoral')
                    
                    # Output projection (row parallel)
                    for gpu in range(8):
                        gpu_id = stage * 8 + gpu
                        mha_cluster.node(f'out_proj_{layer_id}_{gpu}', 
                                       f'Output Proj\n[1024, 4096]\nGPU {gpu_id}',
                                       fillcolor='lightyellow')
                
                # Residual connection
                layer_cluster.node(f'residual1_{layer_id}', 
                                 f'Residual Add\n[1024, 4096]\nAll GPUs',
                                 shape='parallelogram', fillcolor='lightgreen')
                
                # MoE layer - 4 experts per GPU
                with layer_cluster.subgraph(name=f'cluster_moe_{layer_id}') as moe_cluster:
                    moe_cluster.attr(label=f'MoE Layer - 4 Experts per GPU', style='dotted')
                    
                    # Gate computation
                    for gpu in range(8):
                        gpu_id = stage * 8 + gpu
                        moe_cluster.node(f'gate_{layer_id}_{gpu}', 
                                       f'Gate\n[1024, 16]\nGPU {gpu_id}',
                                       shape='parallelogram', fillcolor='orange')
                    
                    # Expert computation - 4 experts per GPU
                    for gpu in range(8):
                        gpu_id = stage * 8 + gpu
                        for expert in range(4):
                            expert_id = gpu * 4 + expert + layer_id * 32
                            moe_cluster.node(f'expert_{layer_id}_{gpu}_{expert}', 
                                           f'Expert {expert_id}\n[256, 32768, 256]\nGPU {gpu_id}',
                                           fillcolor='lightpink')
                    
                    # Expert aggregation
                    for gpu in range(8):
                        gpu_id = stage * 8 + gpu
                        moe_cluster.node(f'agg_{layer_id}_{gpu}', 
                                       f'Expert Agg\n[1024, 4096]\nGPU {gpu_id}',
                                       shape='parallelogram', fillcolor='lightgreen')
                
                # Final residual
                layer_cluster.node(f'residual2_{layer_id}', 
                                 f'Residual Add\n[1024, 4096]\nAll GPUs',
                                 shape='parallelogram', fillcolor='lightgreen')

# Output node
dot.node('output', 'Output\n[1024, hidden_size]\nAll GPUs', shape='ellipse', fillcolor='lightgreen')

# Connect the graph
# Input to first layer
for gpu in range(8):
    dot.edge('input', f'qkv_0_{gpu}')

# Layer connections
for layer_id in range(4):
    stage = layer_id // 2
    
    # MHA connections
    for gpu in range(8):
        gpu_id = stage * 8 + gpu
        dot.edge(f'qkv_{layer_id}_{gpu}', f'attn_{layer_id}_{gpu}')
        dot.edge(f'attn_{layer_id}_{gpu}', f'out_proj_{layer_id}_{gpu}')
        
        if layer_id == 0:
            dot.edge('input', f'residual1_{layer_id}')
        else:
            prev_stage = (layer_id - 1) // 2
            for prev_gpu in range(8):
                prev_gpu_id = prev_stage * 8 + prev_gpu
                dot.edge(f'residual2_{layer_id-1}', f'residual1_{layer_id}', 
                        style='dashed', label='pipeline transfer')
        
        dot.edge(f'out_proj_{layer_id}_{gpu}', f'residual1_{layer_id}')
        
        # MoE connections
        dot.edge(f'residual1_{layer_id}', f'gate_{layer_id}_{gpu}')
        
        # Expert routing (dashed lines)
        for expert in range(4):
            dot.edge(f'gate_{layer_id}_{gpu}', f'expert_{layer_id}_{gpu}_{expert}', 
                    style='dashed', label='route')
            dot.edge(f'expert_{layer_id}_{gpu}_{expert}', f'agg_{layer_id}_{gpu}')
        
        dot.edge(f'agg_{layer_id}_{gpu}', f'residual2_{layer_id}')
        dot.edge(f'residual1_{layer_id}', f'residual2_{layer_id}')

# Final output
for gpu in range(8):
    final_stage = 1
    final_gpu = final_stage * 8 + gpu
    dot.edge(f'residual2_3', 'output')

# Save the DAG
dot.format = 'svg'
dot.render('/home/wzc/data/file-share/2025-09-03-15-36-24/baseline_moe_dag')

# Also save as DOT file
with open('/home/wzc/data/file-share/2025-09-03-15-36-24/baseline_moe_dag.dot', 'w') as f:
    f.write(dot.source)

print("Baseline DAG generated successfully!")
print("Files saved:")
print("- baseline_moe_dag.svg")
print("- baseline_moe_dag.dot")