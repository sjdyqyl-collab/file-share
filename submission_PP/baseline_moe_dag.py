import graphviz

# Create baseline DAG for MoE model with TP=8, PP=2
dot = graphviz.Digraph('baseline_moe', comment='MoE Model - Baseline TP=8, PP=2')
dot.attr(rankdir='TB', size='25,25')

# Define colors for different GPU groups
colors = {
    'tp_group_0': '#ffcccc',   # Light red
    'tp_group_1': '#ccffcc',   # Light green
    'tp_group_2': '#ccccff',   # Light blue
    'tp_group_3': '#ffffcc',   # Light yellow
    'tp_group_4': '#ffccff',   # Light magenta
    'tp_group_5': '#ccffff',   # Light cyan
    'tp_group_6': '#ffddcc',   # Light orange
    'tp_group_7': '#ddffcc',   # Light lime
}

# Input node
dot.node('input', 'Input\n[1024, seq_len, 8192]\nAll GPUs', shape='ellipse', style='filled', fillcolor='lightgray')

# Pipeline Stage 0 (Layers 1-8) on GPUs 0-7
for layer in range(1, 9):
    # Multi-Head Attention
    dot.node(f'layer{layer}_mha_q', f'Layer{layer}\nMHA Q Linear\n[1024, seq_len, 8192]->[1024, seq_len, 8192]\nTP Group 0 (GPUs 0-7)', fillcolor=colors['tp_group_0'])
    dot.node(f'layer{layer}_mha_k', f'Layer{layer}\nMHA K Linear\n[1024, seq_len, 8192]->[1024, seq_len, 8192]\nTP Group 0 (GPUs 0-7)', fillcolor=colors['tp_group_0'])
    dot.node(f'layer{layer}_mha_v', f'Layer{layer}\nMHA V Linear\n[1024, seq_len, 8192]->[1024, seq_len, 8192]\nTP Group 0 (GPUs 0-7)', fillcolor=colors['tp_group_0'])
    dot.node(f'layer{layer}_mha_attn', f'Layer{layer}\nMHA Attention\n[1024, seq_len, 8192]->[1024, seq_len, 8192]\nTP Group 0 (GPUs 0-7)', fillcolor=colors['tp_group_0'])
    dot.node(f'layer{layer}_mha_out', f'Layer{layer}\nMHA Output Linear\n[1024, seq_len, 8192]->[1024, seq_len, 8192]\nTP Group 0 (GPUs 0-7)', fillcolor=colors['tp_group_0'])
    dot.node(f'layer{layer}_mha_add', f'Layer{layer}\nMHA Residual Add\n[1024, seq_len, 8192]\nTP Group 0 (GPUs 0-7)', shape='parallelogram', fillcolor=colors['tp_group_0'])
    
    # MoE Gate
    dot.node(f'layer{layer}_gate', f'Layer{layer}\nMoE Gate\n[1024, seq_len, 8192]->[1024, seq_len, 8]\nTP Group 0 (GPUs 0-7)', shape='diamond', fillcolor=colors['tp_group_0'])
    
    # MoE Experts (8 experts, each on different GPUs within TP group)
    for expert in range(8):
        dot.node(f'layer{layer}_expert{expert}_ffn1', f'Layer{layer}\nExpert{expert} FFN1\n[1024, seq_len, 8192]->[1024, seq_len, 32768]\nGPU {expert}', fillcolor=colors[f'tp_group_{expert}'])
        dot.node(f'layer{layer}_expert{expert}_ffn2', f'Layer{layer}\nExpert{expert} FFN2\n[1024, seq_len, 32768]->[1024, seq_len, 8192]\nGPU {expert}', fillcolor=colors[f'tp_group_{expert}'])
    
    # MoE Aggregation
    dot.node(f'layer{layer}_moe_agg', f'Layer{layer}\nMoE Aggregation\n[1024, seq_len, 8192]\nTP Group 0 (GPUs 0-7)', shape='parallelogram', fillcolor=colors['tp_group_0'])
    dot.node(f'layer{layer}_moe_add', f'Layer{layer}\nMoE Residual Add\n[1024, seq_len, 8192]\nTP Group 0 (GPUs 0-7)', shape='parallelogram', fillcolor=colors['tp_group_0'])
    
    # LayerNorm nodes
    dot.node(f'layer{layer}_ln1', f'Layer{layer}\nLayerNorm 1\n[1024, seq_len, 8192]\nTP Group 0 (GPUs 0-7)', fillcolor=colors['tp_group_0'])
    dot.node(f'layer{layer}_ln2', f'Layer{layer}\nLayerNorm 2\n[1024, seq_len, 8192]\nTP Group 0 (GPUs 0-7)', fillcolor=colors['tp_group_0'])

# Pipeline Stage 1 (Layers 9-16) on GPUs 8-15
for layer in range(9, 17):
    tp_group = (layer - 9) % 8
    color_key = f'tp_group_{tp_group}'
    gpu_start = 8
    
    # Multi-Head Attention
    dot.node(f'layer{layer}_mha_q', f'Layer{layer}\nMHA Q Linear\n[1024, seq_len, 8192]->[1024, seq_len, 8192]\nTP Group {tp_group} (GPUs {gpu_start+tp_group}-{gpu_start+tp_group+7})', fillcolor=colors[f'tp_group_{tp_group}'])
    dot.node(f'layer{layer}_mha_k', f'Layer{layer}\nMHA K Linear\n[1024, seq_len, 8192]->[1024, seq_len, 8192]\nTP Group {tp_group} (GPUs {gpu_start+tp_group}-{gpu_start+tp_group+7})', fillcolor=colors[f'tp_group_{tp_group}'])
    dot.node(f'layer{layer}_mha_v', f'Layer{layer}\nMHA V Linear\n[1024, seq_len, 8192]->[1024, seq_len, 8192]\nTP Group {tp_group} (GPUs {gpu_start+tp_group}-{gpu_start+tp_group+7})', fillcolor=colors[f'tp_group_{tp_group}'])
    dot.node(f'layer{layer}_mha_attn', f'Layer{layer}\nMHA Attention\n[1024, seq_len, 8192]->[1024, seq_len, 8192]\nTP Group {tp_group} (GPUs {gpu_start+tp_group}-{gpu_start+tp_group+7})', fillcolor=colors[f'tp_group_{tp_group}'])
    dot.node(f'layer{layer}_mha_out', f'Layer{layer}\nMHA Output Linear\n[1024, seq_len, 8192]->[1024, seq_len, 8192]\nTP Group {tp_group} (GPUs {gpu_start+tp_group}-{gpu_start+tp_group+7})', fillcolor=colors[f'tp_group_{tp_group}'])
    dot.node(f'layer{layer}_mha_add', f'Layer{layer}\nMHA Residual Add\n[1024, seq_len, 8192]\nTP Group {tp_group} (GPUs {gpu_start+tp_group}-{gpu_start+tp_group+7})', shape='parallelogram', fillcolor=colors[f'tp_group_{tp_group}'])
    
    # MoE Gate
    dot.node(f'layer{layer}_gate', f'Layer{layer}\nMoE Gate\n[1024, seq_len, 8192]->[1024, seq_len, 8]\nTP Group {tp_group} (GPUs {gpu_start+tp_group}-{gpu_start+tp_group+7})', shape='diamond', fillcolor=colors[f'tp_group_{tp_group}'])
    
    # MoE Experts (8 experts, each on different GPUs within TP group)
    for expert in range(8):
        gpu_num = gpu_start + (tp_group + expert) % 8
        dot.node(f'layer{layer}_expert{expert}_ffn1', f'Layer{layer}\nExpert{expert} FFN1\n[1024, seq_len, 8192]->[1024, seq_len, 32768]\nGPU {gpu_num}', fillcolor=colors[f'tp_group_{(tp_group+expert)%8}'])
        dot.node(f'layer{layer}_expert{expert}_ffn2', f'Layer{layer}\nExpert{expert} FFN2\n[1024, seq_len, 32768]->[1024, seq_len, 8192]\nGPU {gpu_num}', fillcolor=colors[f'tp_group_{(tp_group+expert)%8}'])
    
    # MoE Aggregation
    dot.node(f'layer{layer}_moe_agg', f'Layer{layer}\nMoE Aggregation\n[1024, seq_len, 8192]\nTP Group {tp_group} (GPUs {gpu_start+tp_group}-{gpu_start+tp_group+7})', shape='parallelogram', fillcolor=colors[f'tp_group_{tp_group}'])
    dot.node(f'layer{layer}_moe_add', f'Layer{layer}\nMoE Residual Add\n[1024, seq_len, 8192]\nTP Group {tp_group} (GPUs {gpu_start+tp_group}-{gpu_start+tp_group+7})', shape='parallelogram', fillcolor=colors[f'tp_group_{tp_group}'])
    
    # LayerNorm nodes
    dot.node(f'layer{layer}_ln1', f'Layer{layer}\nLayerNorm 1\n[1024, seq_len, 8192]\nTP Group {tp_group} (GPUs {gpu_start+tp_group}-{gpu_start+tp_group+7})', fillcolor=colors[f'tp_group_{tp_group}'])
    dot.node(f'layer{layer}_ln2', f'Layer{layer}\nLayerNorm 2\n[1024, seq_len, 8192]\nTP Group {tp_group} (GPUs {gpu_start+tp_group}-{gpu_start+tp_group+7})', fillcolor=colors[f'tp_group_{tp_group}'])

# Communication nodes for pipeline parallelism
dot.node('pipe_comm_1', 'Pipeline Communication\n[1024, seq_len, 8192]\nGPUs 7 <-> GPUs 8', shape='ellipse', style='filled', fillcolor='orange')

# Output node
dot.node('output', 'Output\n[1024, seq_len, 8192]\nAll GPUs', shape='ellipse', style='filled', fillcolor='lightgray')

# Connect the DAG
# Input to first layer
dot.edge('input', 'layer1_ln1')

for layer in range(1, 17):
    # MHA path
    dot.edge(f'layer{layer}_ln1', f'layer{layer}_mha_q')
    dot.edge(f'layer{layer}_ln1', f'layer{layer}_mha_k')
    dot.edge(f'layer{layer}_ln1', f'layer{layer}_mha_v')
    dot.edge(f'layer{layer}_mha_q', f'layer{layer}_mha_attn')
    dot.edge(f'layer{layer}_mha_k', f'layer{layer}_mha_attn')
    dot.edge(f'layer{layer}_mha_v', f'layer{layer}_mha_attn')
    dot.edge(f'layer{layer}_mha_attn', f'layer{layer}_mha_out')
    dot.edge(f'layer{layer}_mha_out', f'layer{layer}_mha_add')
    
    # Add residual connection
    if layer == 1:
        dot.edge('input', f'layer1_mha_add')
    else:
        if layer <= 8:
            dot.edge(f'layer{layer-1}_moe_add', f'layer{layer}_mha_add')
        else:
            if layer == 9:
                dot.edge('pipe_comm_1', f'layer9_mha_add')
            else:
                dot.edge(f'layer{layer-1}_moe_add', f'layer{layer}_mha_add')
    
    dot.edge(f'layer{layer}_mha_add', f'layer{layer}_ln2')
    
    # MoE path
    dot.edge(f'layer{layer}_ln2', f'layer{layer}_gate')
    
    # Connect gate to experts (dashed lines for routing)
    for expert in range(8):
        dot.edge(f'layer{layer}_gate', f'layer{layer}_expert{expert}_ffn1', style='dashed')
        dot.edge(f'layer{layer}_ln2', f'layer{layer}_expert{expert}_ffn1', style='dashed')
        dot.edge(f'layer{layer}_expert{expert}_ffn1', f'layer{layer}_expert{expert}_ffn2')
        dot.edge(f'layer{layer}_expert{expert}_ffn2', f'layer{layer}_moe_agg')
    
    dot.edge(f'layer{layer}_moe_agg', f'layer{layer}_moe_add')
    dot.edge(f'layer{layer}_mha_add', f'layer{layer}_moe_add')
    
    # Connect to next layer or output
    if layer == 8:
        dot.edge(f'layer{layer}_moe_add', 'pipe_comm_1')
    elif layer == 16:
        dot.edge(f'layer{layer}_moe_add', 'output')
    else:
        dot.edge(f'layer{layer}_moe_add', f'layer{layer+1}_ln1')

# Save the DAG
dot.format = 'svg'
dot.render('/home/wzc/data/file-share/submission/baseline_moe_dag', cleanup=True)

print("Baseline MoE DAG generated successfully")