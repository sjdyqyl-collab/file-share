#!/usr/bin/env python3
import graphviz

# Create proposed DAG with 64 GPUs (EP=64)
# Each GPU has exactly 1 expert

dot = graphviz.Digraph('proposed_moe_dag', comment='Proposed MoE with EP=64, 64 GPUs (1 expert per GPU)')
dot.attr(rankdir='TB', size='30,40')

# Define node styles
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
dot.attr('edge', fontsize='10')

# Input node
dot.node('input', 'Input\n[1024, hidden_size]\nAll GPUs', shape='ellipse', fillcolor='lightgreen')

# Since we have 64 GPUs and 64 experts total (16 per layer), we need to distribute:
# - Layer 0: Experts 0-15 on GPUs 0-15
# - Layer 1: Experts 16-31 on GPUs 16-31
# - Layer 2: Experts 32-47 on GPUs 32-47
# - Layer 3: Experts 48-63 on GPUs 48-63

for layer in range(4):
    with dot.subgraph(name=f'cluster_layer_{layer}') as layer_cluster:
        layer_cluster.attr(label=f'Layer {layer} (Experts {layer*16}-{(layer+1)*16-1})', style='dashed')
        
        # MHA - since we don't have TP specified, let's assume MHA is replicated
        # or uses a different parallelism strategy
        with layer_cluster.subgraph(name=f'cluster_mha_{layer}') as mha_cluster:
            mha_cluster.attr(label='Multi-Head Attention', style='dotted')
            
            # Input preprocessing (all GPUs in this layer's range)
            for gpu in range(layer*16, (layer+1)*16):
                mha_cluster.node(f'preprocess_{layer}_{gpu}', 
                               f'Input Process\n[1024, 4096]\nGPU {gpu}',
                               fillcolor='lightyellow')
            
            # QKV projection (distributed)
            for gpu in range(layer*16, (layer+1)*16):
                mha_cluster.node(f'qkv_{layer}_{gpu}', 
                               f'QKV Proj\n[1024, 768]\nGPU {gpu}',
                               fillcolor='lightyellow')
            
            # Attention heads (16 heads × 512 = 8192, distributed across 16 GPUs)
            for gpu in range(layer*16, (layer+1)*16):
                mha_cluster.node(f'attn_{layer}_{gpu}', 
                               f'Attention\n[64, 512]\nGPU {gpu}',
                               fillcolor='lightcoral')
            
            # Output projection
            for gpu in range(layer*16, (layer+1)*16):
                mha_cluster.node(f'out_proj_{layer}_{gpu}', 
                               f'Output Proj\n[1024, 4096]\nGPU {gpu}',
                               fillcolor='lightyellow')
        
        # Residual connection 1
        layer_cluster.node(f'residual1_{layer}', 
                         f'Residual Add\n[1024, 4096]\nLayer {layer} GPUs',
                         shape='parallelogram', fillcolor='lightgreen')
        
        # MoE layer - 1 expert per GPU
        with layer_cluster.subgraph(name=f'cluster_moe_{layer}') as moe_cluster:
            moe_cluster.attr(label=f'MoE Layer - 1 Expert per GPU', style='dotted')
            
            # Gate computation - distributed across all 16 GPUs in layer
            for gpu in range(layer*16, (layer+1)*16):
                moe_cluster.node(f'gate_{layer}_{gpu}', 
                               f'Gate\n[1024, 16]\nGPU {gpu}',
                               shape='parallelogram', fillcolor='orange')
            
            # All-to-all communication for token routing
            moe_cluster.node(f'all2all_{layer}_send', 
                           f'All2All Send\n[Variable, 4096]\nAll Layer GPUs',
                           shape='ellipse', fillcolor='lightblue')
            
            # Expert computation - exactly 1 expert per GPU
            for gpu in range(layer*16, (layer+1)*16):
                expert_id = gpu  # Each GPU has exactly one expert
                moe_cluster.node(f'expert_{layer}_{gpu}', 
                               f'Expert {expert_id}\n[Variable, 32768, 4096]\nGPU {gpu}',
                               fillcolor='lightpink')
            
            # All-to-all communication for token aggregation
            moe_cluster.node(f'all2all_{layer}_recv', 
                           f'All2All Recv\n[1024, 4096]\nAll Layer GPUs',
                           shape='ellipse', fillcolor='lightblue')
            
            # Expert aggregation
            for gpu in range(layer*16, (layer+1)*16):
                moe_cluster.node(f'agg_{layer}_{gpu}', 
                               f'Expert Agg\n[1024, 4096]\nGPU {gpu}',
                               shape='parallelogram', fillcolor='lightgreen')
        
        # Final residual
        layer_cluster.node(f'residual2_{layer}', 
                         f'Residual Add\n[1024, 4096]\nLayer {layer} GPUs',
                         shape='parallelogram', fillcolor='lightgreen')

# Output node
dot.node('output', 'Output\n[1024, hidden_size]\nAll GPUs', shape='ellipse', fillcolor='lightgreen')

# Connect the graph
# Input to first layer
for gpu in range(16):
    dot.edge('input', f'preprocess_0_{gpu}')

# Layer connections
for layer in range(4):
    # MHA connections
    for gpu in range(layer*16, (layer+1)*16):
        dot.edge(f'preprocess_{layer}_{gpu}', f'qkv_{layer}_{gpu}')
        dot.edge(f'qkv_{layer}_{gpu}', f'attn_{layer}_{gpu}')
        dot.edge(f'attn_{layer}_{gpu}', f'out_proj_{layer}_{gpu}')
        
        if layer == 0:
            dot.edge('input', f'residual1_{layer}')
        else:
            # Connect from previous layer
            for prev_gpu in range((layer-1)*16, layer*16):
                dot.edge(f'residual2_{layer-1}', f'preprocess_{layer}_{gpu}',
                        style='dashed', label='cross-layer')
        
        dot.edge(f'out_proj_{layer}_{gpu}', f'residual1_{layer}')
        
        # MoE connections
        dot.edge(f'residual1_{layer}', f'gate_{layer}_{gpu}')
        dot.edge(f'gate_{layer}_{gpu}', f'all2all_{layer}_send', 
                style='dashed', label='routing')
        
        # All-to-all routing
        dot.edge(f'all2all_{layer}_send', f'expert_{layer}_{gpu}',
                style='dashed', label='token routing')
        dot.edge(f'expert_{layer}_{gpu}', f'all2all_{layer}_recv')
        dot.edge(f'all2all_{layer}_recv', f'agg_{layer}_{gpu}')
        dot.edge(f'agg_{layer}_{gpu}', f'residual2_{layer}')
        dot.edge(f'residual1_{layer}', f'residual2_{layer}')

# Final output
for gpu in range(48, 64):  # Last layer GPUs
    dot.edge(f'residual2_3', 'output')

# Save the DAG
dot.format = 'svg'
dot.render('/home/wzc/data/file-share/2025-09-03-15-36-24/proposed_moe_dag')

# Also save as DOT file
with open('/home/wzc/data/file-share/2025-09-03-15-36-24/proposed_moe_dag.dot', 'w') as f:
    f.write(dot.source)

print("Proposed DAG generated successfully!")
print("Files saved:")
print("- proposed_moe_dag.svg")
print("- proposed_moe_dag.dot")