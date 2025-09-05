#!/usr/bin/env python3

import graphviz

def create_baseline_dag():
    dot = graphviz.Digraph('Baseline_MoE_TP8_PP2')
    dot.attr(rankdir='TB', size='20,30')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Define GPU clusters for pipeline stages
    with dot.subgraph(name='cluster_pipeline_stage_0') as c0:
        c0.attr(label='Pipeline Stage 0 (GPUs 0-7)', style='dashed', color='red')
        
        # Layer 0 - Stage 0 - MHA part
        c0.node('mha_0_qkv', 'MHA QKV Projection\nTP=8 across GPUs 0-7\nInput: [1024, 8192]\nOutput: [1024, 8192]', shape='ellipse', fillcolor='lightgreen')
        c0.node('mha_0_attn', 'MHA Attention\nTP=8 across GPUs 0-7\nInput: [1024, 1024]\nOutput: [1024, 1024]', shape='ellipse', fillcolor='yellow')
        c0.node('mha_0_out', 'MHA Output Projection\nTP=8 across GPUs 0-7\nInput: [1024, 1024]\nOutput: [1024, 8192]', shape='ellipse', fillcolor='lightgreen')
        c0.node('res_0', 'Residual Add\nAll GPUs 0-7\nInput: [1024, 8192] × 2\nOutput: [1024, 8192]', shape='parallelogram', fillcolor='orange')
        
        # Experts for stage 0 (experts 0-7, 2 per GPU)
        for gpu in range(8):
            for exp in range(2):
                expert_id = gpu * 2 + exp
                c0.node(f'exp_0_{expert_id}', f'MoE Expert {expert_id}\nGPU {gpu}\nInput: [1024, 8192]\nOutput: [1024, 8192]', shape='rectangle', fillcolor='pink')
        
        c0.node('gate_0', 'MoE Gate\nAll GPUs 0-7\nInput: [1024, 8192]\nOutput: [1024, 16] routing', shape='diamond', fillcolor='purple', style='dashed')
        c0.node('agg_0', 'Expert Aggregation\nAll GPUs 0-7\nInput: [1024, 8192] × 16\nOutput: [1024, 8192]', shape='parallelogram', fillcolor='orange')
        c0.node('res2_0', 'Residual Add\nAll GPUs 0-7\nInput: [1024, 8192] × 2\nOutput: [1024, 8192]', shape='parallelogram', fillcolor='orange')
    
    with dot.subgraph(name='cluster_pipeline_stage_1') as c1:
        c1.attr(label='Pipeline Stage 1 (GPUs 8-15)', style='dashed', color='blue')
        
        # Layer 0 - Stage 1 - Experts 8-15
        for gpu in range(8, 16):
            for exp in range(2):
                expert_id = (gpu - 8) * 2 + exp + 8
                c1.node(f'exp_0_{expert_id}', f'MoE Expert {expert_id}\nGPU {gpu}\nInput: [1024, 8192]\nOutput: [1024, 8192]', shape='rectangle', fillcolor='pink')
    
    # Repeat for all 4 layers
    for layer in range(1, 4):
        with dot.subgraph(name=f'cluster_layer_{layer}_stage_0') as cl0:
            cl0.attr(label=f'Layer {layer} Stage 0', style='dotted')
            cl0.node(f'mha_{layer}_qkv', f'MHA QKV Layer {layer}\nTP=8 GPUs 0-7\nInput: [1024, 8192]\nOutput: [1024, 8192]', shape='ellipse', fillcolor='lightgreen')
            cl0.node(f'mha_{layer}_attn', f'MHA Attention Layer {layer}\nTP=8 GPUs 0-7\nInput: [1024, 1024]\nOutput: [1024, 1024]', shape='ellipse', fillcolor='yellow')
            cl0.node(f'mha_{layer}_out', f'MHA Output Layer {layer}\nTP=8 GPUs 0-7\nInput: [1024, 1024]\nOutput: [1024, 8192]', shape='ellipse', fillcolor='lightgreen')
            cl0.node(f'res_{layer}', f'Residual Add Layer {layer}\nAll GPUs\nInput: [1024, 8192] × 2\nOutput: [1024, 8192]', shape='parallelogram', fillcolor='orange')
            
            for gpu in range(8):
                for exp in range(2):
                    expert_id = gpu * 2 + exp
                    cl0.node(f'exp_{layer}_{expert_id}', f'MoE Expert {expert_id}\nGPU {gpu}\nLayer {layer}\nInput: [1024, 8192]\nOutput: [1024, 8192]', shape='rectangle', fillcolor='pink')
            
            cl0.node(f'gate_{layer}', f'MoE Gate Layer {layer}\nAll GPUs\nInput: [1024, 8192]\nOutput: [1024, 16] routing', shape='diamond', fillcolor='purple', style='dashed')
            cl0.node(f'agg_{layer}', f'Expert Aggregation Layer {layer}\nAll GPUs\nInput: [1024, 8192] × 16\nOutput: [1024, 8192]', shape='parallelogram', fillcolor='orange')
            cl0.node(f'res2_{layer}', f'Residual Add Layer {layer}\nAll GPUs\nInput: [1024, 8192] × 2\nOutput: [1024, 8192]', shape='parallelogram', fillcolor='orange')
    
    # Input and output
    dot.node('input', 'Model Input\n[1024, 8192]\nBatch Size: 1024 tokens', shape='ellipse', fillcolor='white')
    dot.node('output', 'Model Output\n[1024, 8192]', shape='ellipse', fillcolor='white')
    
    # Pipeline communication
    dot.node('pipe_comm_0_1', 'Pipeline Communication\nStage 0 → Stage 1\n[1024, 8192]', shape='parallelogram', fillcolor='red', style='dashed')
    dot.node('pipe_comm_1_0', 'Pipeline Communication\nStage 1 → Stage 0\n[1024, 8192]', shape='parallelogram', fillcolor='red', style='dashed')
    
    # Connections
    dot.edge('input', 'mha_0_qkv')
    dot.edge('mha_0_qkv', 'mha_0_attn')
    dot.edge('mha_0_attn', 'mha_0_out')
    dot.edge('mha_0_out', 'res_0')
    dot.edge('res_0', 'gate_0')
    
    for i in range(16):
        dot.edge('gate_0', f'exp_0_{i}')
        dot.edge(f'exp_0_{i}', 'agg_0')
    
    dot.edge('agg_0', 'res2_0')
    dot.edge('res2_0', 'mha_1_qkv')
    
    # Continue for all layers...
    for layer in range(1, 4):
        dot.edge(f'mha_{layer}_qkv', f'mha_{layer}_attn')
        dot.edge(f'mha_{layer}_attn', f'mha_{layer}_out')
        dot.edge(f'mha_{layer}_out', f'res_{layer}')
        dot.edge(f'res_{layer}', f'gate_{layer}')
        
        for i in range(16):
            dot.edge(f'gate_{layer}', f'exp_{layer}_{i}')
            dot.edge(f'exp_{layer}_{i}', f'agg_{layer}')
        
        dot.edge(f'agg_{layer}', f'res2_{layer}')
        if layer == 3:
            dot.edge(f'res2_{layer}', 'output')
        else:
            dot.edge(f'res2_{layer}', f'mha_{layer+1}_qkv')
    
    # Save files
    with open('/home/wzc/data/file-share/2025-09-04-19-24-34/baseline_moe_dag.dot', 'w') as f:
        f.write(dot.source)
    
    dot.format = 'svg'
    dot.render('/home/wzc/data/file-share/2025-09-04-19-24-34/baseline_moe_dag', cleanup=False)

if __name__ == '__main__':
    create_baseline_dag()