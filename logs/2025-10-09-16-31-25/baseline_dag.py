#!/usr/bin/env python3

import graphviz

def create_baseline_dag():
    dot = graphviz.Digraph('Baseline_Architecture', 
                          comment='Baseline: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)',
                          format='svg')
    
    # Set graph attributes
    dot.attr(rankdir='TB', size='30,20')
    dot.attr('node', shape='rectangle', style='filled', fontname='Arial')
    
    # Color scheme
    tp_color = 'lightblue'  # Tensor parallelism
    pp_color = 'lightgreen'  # Pipeline parallelism
    communication_color = 'lightyellow'
    routing_color = 'lightcoral'
    aggregation_color = 'lightpink'
    
    # Pipeline Stage 1 - Layers 1-2 (GPUs 0-7)
    with dot.subgraph(name='cluster_pipeline_stage1') as c:
        c.attr(label='Pipeline Stage 1 - Layers 1-2 (GPUs 0-7)', style='dashed', color='black')
        
        # Input for stage 1
        c.node('input', 'Model Input\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\nGPU: 0-7',
               shape='ellipse', fillcolor='lightgray')
        
        # Layer 1 - Tensor Parallel
        with c.subgraph(name='cluster_layer1_tp') as tp1:
            tp1.attr(label='Layer 1 - Tensor Parallel (TP=8)', style='dotted')
            
            # QKV projections split across 8 GPUs
            for gpu in range(8):
                tp1.node(f'l1_qkv_tp{gpu}', f'QKV Projection Split\nGPU {gpu}\nInput: [1024, 2048, 512]\nOutput: [1024, 2048, 4, 64]×3\nGPU: {gpu}',
                        fillcolor=tp_color)
            
            # Attention computation split across 8 GPUs
            for gpu in range(8):
                tp1.node(f'l1_attn_tp{gpu}', f'Attention Split\nGPU {gpu}\nInput: Q[1024,2048,4,64], K[1024,2048,4,64], V[1024,2048,4,64]\nOutput: [1024,2048,4,64]\nGPU: {gpu}',
                        fillcolor=tp_color)
            
            # All-reduce for attention output
            tp1.node('l1_attn_allreduce', 'All-Reduce Attention\nInput: [1024,2048,4,64]×8\nOutput: [1024,2048,32,64]\nGPU: 0-7',
                    shape='parallelogram', fillcolor=communication_color)
            
            # FFN/MoE split across 8 GPUs
            for gpu in range(8):
                tp1.node(f'l1_ffn_tp{gpu}', f'FFN/MoE Split\nGPU {gpu}\nInput: [1024, 2048, 512]\nOutput: [1024, 2048, 512]\nGPU: {gpu}',
                        fillcolor=tp_color)
            
            # All-reduce for FFN output
            tp1.node('l1_ffn_allreduce', 'All-Reduce FFN\nInput: [1024,2048,512]×8\nOutput: [1024,2048,4096]\nGPU: 0-7',
                    shape='parallelogram', fillcolor=communication_color)
        
        # Layer 2 - Tensor Parallel
        with c.subgraph(name='cluster_layer2_tp') as tp2:
            tp2.attr(label='Layer 2 - Tensor Parallel (TP=8)', style='dotted')
            
            for gpu in range(8):
                tp2.node(f'l2_qkv_tp{gpu}', f'QKV Projection Split\nGPU {gpu}\nInput: [1024, 2048, 512]\nOutput: [1024, 2048, 4, 64]×3\nGPU: {gpu}',
                        fillcolor=tp_color)
            
            for gpu in range(8):
                tp2.node(f'l2_attn_tp{gpu}', f'Attention Split\nGPU {gpu}\nInput: Q[1024,2048,4,64], K[1024,2048,4,64], V[1024,2048,4,64]\nOutput: [1024,2048,4,64]\nGPU: {gpu}',
                        fillcolor=tp_color)
            
            tp2.node('l2_attn_allreduce', 'All-Reduce Attention\nInput: [1024,2048,4,64]×8\nOutput: [1024,2048,32,64]\nGPU: 0-7',
                    shape='parallelogram', fillcolor=communication_color)
            
            for gpu in range(8):
                tp2.node(f'l2_ffn_tp{gpu}', f'FFN/MoE Split\nGPU {gpu}\nInput: [1024, 2048, 512]\nOutput: [1024, 2048, 512]\nGPU: {gpu}',
                        fillcolor=tp_color)
            
            tp2.node('l2_ffn_allreduce', 'All-Reduce FFN\nInput: [1024,2048,512]×8\nOutput: [1024,2048,4096]\nGPU: 0-7',
                    shape='parallelogram', fillcolor=communication_color)
        
        # Residual connections
        c.node('l1_residual', 'Residual Add\nInput1: [1024,2048,4096] (input)\nInput2: [1024,2048,4096] (layer1 output)\nOutput: [1024,2048,4096]\nGPU: 0-7',
                fillcolor='lightgray')
        c.node('l2_residual', 'Residual Add\nInput1: [1024,2048,4096] (layer1 output)\nInput2: [1024,2048,4096] (layer2 output)\nOutput: [1024,2048,4096]\nGPU: 0-7',
                fillcolor='lightgray')
    
    # Pipeline Stage 2 - Layers 3-4 (GPUs 8-15)
    with dot.subgraph(name='cluster_pipeline_stage2') as c:
        c.attr(label='Pipeline Stage 2 - Layers 3-4 (GPUs 8-15)', style='dashed', color='black')
        
        # Pipeline communication
        c.node('pipeline_comm', 'Pipeline Communication\nInput: [1024,2048,4096] from Stage 1\nOutput: [1024,2048,4096] to Stage 2\nGPU: 0-7 → 8-15',
               shape='parallelogram', fillcolor=communication_color)
        
        # Layer 3 - Tensor Parallel
        with c.subgraph(name='cluster_layer3_tp') as tp3:
            tp3.attr(label='Layer 3 - Tensor Parallel (TP=8)', style='dotted')
            
            for gpu in range(8):
                tp3.node(f'l3_qkv_tp{gpu+8}', f'QKV Projection Split\nGPU {gpu+8}\nInput: [1024, 2048, 512]\nOutput: [1024, 2048, 4, 64]×3\nGPU: {gpu+8}',
                        fillcolor=pp_color)
            
            for gpu in range(8):
                tp3.node(f'l3_attn_tp{gpu+8}', f'Attention Split\nGPU {gpu+8}\nInput: Q[1024,2048,4,64], K[1024,2048,4,64], V[1024,2048,4,64]\nOutput: [1024,2048,4,64]\nGPU: {gpu+8}',
                        fillcolor=pp_color)
            
            tp3.node('l3_attn_allreduce', 'All-Reduce Attention\nInput: [1024,2048,4,64]×8\nOutput: [1024,2048,32,64]\nGPU: 8-15',
                    shape='parallelogram', fillcolor=communication_color)
            
            for gpu in range(8):
                tp3.node(f'l3_ffn_tp{gpu+8}', f'FFN/MoE Split\nGPU {gpu+8}\nInput: [1024, 2048, 512]\nOutput: [1024, 2048, 512]\nGPU: {gpu+8}',
                        fillcolor=pp_color)
            
            tp3.node('l3_ffn_allreduce', 'All-Reduce FFN\nInput: [1024,2048,512]×8\nOutput: [1024,2048,4096]\nGPU: 8-15',
                    shape='parallelogram', fillcolor=communication_color)
        
        # Layer 4 - Tensor Parallel
        with c.subgraph(name='cluster_layer4_tp') as tp4:
            tp4.attr(label='Layer 4 - Tensor Parallel (TP=8)', style='dotted')
            
            for gpu in range(8):
                tp4.node(f'l4_qkv_tp{gpu+8}', f'QKV Projection Split\nGPU {gpu+8}\nInput: [1024, 2048, 512]\nOutput: [1024, 2048, 4, 64]×3\nGPU: {gpu+8}',
                        fillcolor=pp_color)
            
            for gpu in range(8):
                tp4.node(f'l4_attn_tp{gpu+8}', f'Attention Split\nGPU {gpu+8}\nInput: Q[1024,2048,4,64], K[1024,2048,4,64], V[1024,2048,4,64]\nOutput: [1024,2048,4,64]\nGPU: {gpu+8}',
                        fillcolor=pp_color)
            
            tp4.node('l4_attn_allreduce', 'All-Reduce Attention\nInput: [1024,2048,4,64]×8\nOutput: [1024,2048,32,64]\nGPU: 8-15',
                    shape='parallelogram', fillcolor=communication_color)
            
            for gpu in range(8):
                tp4.node(f'l4_ffn_tp{gpu+8}', f'FFN/MoE Split\nGPU {gpu+8}\nInput: [1024, 2048, 512]\nOutput: [1024, 2048, 512]\nGPU: {gpu+8}',
                        fillcolor=pp_color)
            
            tp4.node('l4_ffn_allreduce', 'All-Reduce FFN\nInput: [1024,2048,512]×8\nOutput: [1024,2048,4096]\nGPU: 8-15',
                    shape='parallelogram', fillcolor=communication_color)
        
        # Residual connections
        c.node('l3_residual', 'Residual Add\nInput1: [1024,2048,4096] (stage2 input)\nInput2: [1024,2048,4096] (layer3 output)\nOutput: [1024,2048,4096]\nGPU: 8-15',
                fillcolor='lightgray')
        c.node('l4_residual', 'Residual Add\nInput1: [1024,2048,4096] (layer3 output)\nInput2: [1024,2048,4096] (layer4 output)\nOutput: [1024,2048,4096]\nGPU: 8-15',
                fillcolor='lightgray')
    
    # Output layer
    dot.node('output', 'Model Output\nInput: [1024, 2048, 4096]\nOutput: [1024, 2048, 50265]\nGPU: 8-15',
             shape='ellipse', fillcolor='lightgray')
    
    # Connections for Pipeline Stage 1
    dot.edge('input', 'l1_qkv_tp0')
    dot.edge('input', 'l1_qkv_tp1')
    dot.edge('input', 'l1_qkv_tp2')
    dot.edge('input', 'l1_qkv_tp3')
    dot.edge('input', 'l1_qkv_tp4')
    dot.edge('input', 'l1_qkv_tp5')
    dot.edge('input', 'l1_qkv_tp6')
    dot.edge('input', 'l1_qkv_tp7')
    
    for gpu in range(8):
        dot.edge(f'l1_qkv_tp{gpu}', f'l1_attn_tp{gpu}')
        dot.edge(f'l1_attn_tp{gpu}', 'l1_attn_allreduce')
        dot.edge('l1_attn_allreduce', f'l1_ffn_tp{gpu}')
        dot.edge(f'l1_ffn_tp{gpu}', 'l1_ffn_allreduce')
    
    dot.edge('l1_ffn_allreduce', 'l2_qkv_tp0')
    dot.edge('l1_ffn_allreduce', 'l2_qkv_tp1')
    dot.edge('l1_ffn_allreduce', 'l2_qkv_tp2')
    dot.edge('l1_ffn_allreduce', 'l2_qkv_tp3')
    dot.edge('l1_ffn_allreduce', 'l2_qkv_tp4')
    dot.edge('l1_ffn_allreduce', 'l2_qkv_tp5')
    dot.edge('l1_ffn_allreduce', 'l2_qkv_tp6')
    dot.edge('l1_ffn_allreduce', 'l2_qkv_tp7')
    
    for gpu in range(8):
        dot.edge(f'l2_qkv_tp{gpu}', f'l2_attn_tp{gpu}')
        dot.edge(f'l2_attn_tp{gpu}', 'l2_attn_allreduce')
        dot.edge('l2_attn_allreduce', f'l2_ffn_tp{gpu}')
        dot.edge(f'l2_ffn_tp{gpu}', 'l2_ffn_allreduce')
    
    # Connections for Pipeline Stage 2
    dot.edge('l2_ffn_allreduce', 'pipeline_comm')
    dot.edge('pipeline_comm', 'l3_qkv_tp8')
    dot.edge('pipeline_comm', 'l3_qkv_tp9')
    dot.edge('pipeline_comm', 'l3_qkv_tp10')
    dot.edge('pipeline_comm', 'l3_qkv_tp11')
    dot.edge('pipeline_comm', 'l3_qkv_tp12')
    dot.edge('pipeline_comm', 'l3_qkv_tp13')
    dot.edge('pipeline_comm', 'l3_qkv_tp14')
    dot.edge('pipeline_comm', 'l3_qkv_tp15')
    
    for gpu in range(8):
        dot.edge(f'l3_qkv_tp{gpu+8}', f'l3_attn_tp{gpu+8}')
        dot.edge(f'l3_attn_tp{gpu+8}', 'l3_attn_allreduce')
        dot.edge('l3_attn_allreduce', f'l3_ffn_tp{gpu+8}')
        dot.edge(f'l3_ffn_tp{gpu+8}', 'l3_ffn_allreduce')
    
    dot.edge('l3_ffn_allreduce', 'l4_qkv_tp8')
    dot.edge('l3_ffn_allreduce', 'l4_qkv_tp9')
    dot.edge('l3_ffn_allreduce', 'l4_qkv_tp10')
    dot.edge('l3_ffn_allreduce', 'l4_qkv_tp11')
    dot.edge('l3_ffn_allreduce', 'l4_qkv_tp12')
    dot.edge('l3_ffn_allreduce', 'l4_qkv_tp13')
    dot.edge('l3_ffn_allreduce', 'l4_qkv_tp14')
    dot.edge('l3_ffn_allreduce', 'l4_qkv_tp15')
    
    for gpu in range(8):
        dot.edge(f'l4_qkv_tp{gpu+8}', f'l4_attn_tp{gpu+8}')
        dot.edge(f'l4_attn_tp{gpu+8}', 'l4_attn_allreduce')
        dot.edge('l4_attn_allreduce', f'l4_ffn_tp{gpu+8}')
        dot.edge(f'l4_ffn_tp{gpu+8}', 'l4_ffn_allreduce')
    
    dot.edge('l4_ffn_allreduce', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/logs/2025-10-09-16-31-25/baseline_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/logs/2025-10-09-16-31-25/baseline_dag.dot')