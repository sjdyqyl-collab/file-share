#!/usr/bin/env python3

import graphviz

def generate_baseline_dag():
    """Generate DAG for Traditional MoE Parallelism (Baseline)"""
    
    dot = graphviz.Digraph('baseline_moe_parallelism', format='svg')
    dot.attr(rankdir='TB', size='40,30', ranksep='2.0', nodesep='1.0')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    dot.attr('node', shape='diamond', style='filled', fillcolor='lightcoral')  # Communication
    
    # Global input
    dot.node('input', 'Input\n[1024 seqs, 10000 tokens, 8192 dim]', shape='ellipse', fillcolor='lightblue')
    
    # Model has 4 layers, each with 16 experts
    # 16 GPUs total, each GPU has 4 experts (64 experts / 16 GPUs = 4 experts per GPU)
    # TP=8, PP=2
    
    # Pipeline stage 0: Layers 0-1 on GPUs 0-7
    # Pipeline stage 1: Layers 2-3 on GPUs 8-15
    
    prev_output = 'input'
    
    for layer in range(4):
        # Determine which pipeline stage this layer belongs to
        pipeline_stage = 0 if layer < 2 else 1
        gpu_start = 0 if pipeline_stage == 0 else 8
        
        # Layer norm before attention
        ln1_name = f'layer{layer}_ln1'
        dot.node(ln1_name, f'LayerNorm\n[1024, 10000, 8192]\nAll GPUs {gpu_start}-{gpu_start+7}', 
                shape='rectangle', fillcolor='lightgreen')
        dot.edge(prev_output, ln1_name)
        
        # Multi-head attention (TP=8 across 8 GPUs)
        attn_name = f'layer{layer}_attn'
        dot.node(attn_name, f'Multi-Head Attention\n16 heads × 512 dim\n[1024, 10000, 8192]\nAll GPUs {gpu_start}-{gpu_start+7}', 
                shape='rectangle', fillcolor='lightgreen')
        dot.edge(ln1_name, attn_name)
        
        # Attention output aggregation (TP reduction)
        attn_agg_name = f'layer{layer}_attn_agg'
        dot.node(attn_agg_name, f'All-Reduce\nAttention Output\n[1024, 10000, 8192]\nAll GPUs {gpu_start}-{gpu_start+7}', 
                shape='diamond', fillcolor='lightcoral')
        dot.edge(attn_name, attn_agg_name)
        
        # Residual connection
        residual1_name = f'layer{layer}_residual1'
        dot.node(residual1_name, f'Residual Add\n[1024, 10000, 8192]\nAll GPUs {gpu_start}-{gpu_start+7}', 
                shape='parallelogram', fillcolor='lightyellow')
        dot.edge(prev_output, residual1_name)
        dot.edge(attn_agg_name, residual1_name)
        
        # Layer norm before MoE
        ln2_name = f'layer{layer}_ln2'
        dot.node(ln2_name, f'LayerNorm\n[1024, 10000, 8192]\nAll GPUs {gpu_start}-{gpu_start+7}', 
                shape='rectangle', fillcolor='lightgreen')
        dot.edge(residual1_name, ln2_name)
        
        # Gate for expert selection
        gate_name = f'layer{layer}_gate'
        dot.node(gate_name, f'Gate\nSelect top-2 experts\n[1024, 10000, 16]\nAll GPUs {gpu_start}-{gpu_start+7}', 
                shape='parallelogram', fillcolor='lightyellow')
        dot.edge(ln2_name, gate_name)
        
        # Expert computation - 4 experts per GPU
        expert_outputs = []
        for gpu_id in range(gpu_start, gpu_start + 8):
            gpu_experts_start = (gpu_id - gpu_start) * 4 + layer * 16
            for expert_idx in range(4):
                expert_id = gpu_experts_start + expert_idx
                expert_name = f'layer{layer}_expert{expert_id}_gpu{gpu_id}'
                dot.node(expert_name, f'Expert {expert_id}\nMLP\n[32768 hidden]\nGPU {gpu_id}', 
                        shape='rectangle', fillcolor='lightgreen')
                dot.edge(ln2_name, expert_name)
                
                # Dashed line from gate to expert (routing decision)
                dot.edge(gate_name, expert_name, style='dashed', label=f'expert {expert_id}')
                
                expert_outputs.append(expert_name)
        
        # Expert output aggregation
        expert_agg_name = f'layer{layer}_expert_agg'
        dot.node(expert_agg_name, f'Aggregate Expert Outputs\n[1024, 10000, 8192]\nAll GPUs {gpu_start}-{gpu_start+7}', 
                shape='parallelogram', fillcolor='lightyellow')
        
        for expert_output in expert_outputs:
            dot.edge(expert_output, expert_agg_name)
        
        # Residual connection after MoE
        residual2_name = f'layer{layer}_residual2'
        dot.node(residual2_name, f'Residual Add\n[1024, 10000, 8192]\nAll GPUs {gpu_start}-{gpu_start+7}', 
                shape='parallelogram', fillcolor='lightyellow')
        dot.edge(residual1_name, residual2_name)
        dot.edge(expert_agg_name, residual2_name)
        
        # Pipeline communication between stages
        if layer == 1:
            # Communication from stage 0 to stage 1
            comm_name = f'pipeline_comm_layer{layer}'
            dot.node(comm_name, f'Pipeline Communication\nStage 0 → Stage 1\n[1024, 10000, 8192]\nGPU 7 → GPU 8', 
                    shape='diamond', fillcolor='lightcoral')
            dot.edge(residual2_name, comm_name)
            prev_output = comm_name
        else:
            prev_output = residual2_name
    
    # Global output
    dot.node('output', 'Output\n[1024 seqs, 10000 tokens, 8192 dim]', shape='ellipse', fillcolor='lightblue')
    dot.edge(prev_output, 'output')
    
    # Save files
    dot.render('/home/wzc/data/file-share/2025-09-05-17-32-21/baseline_moe_parallelism', format='svg', cleanup=False)
    
    # Also save as .dot file
    with open('/home/wzc/data/file-share/2025-09-05-17-32-21/baseline_moe_parallelism.dot', 'w') as f:
        f.write(dot.source)
    
    return dot.source

if __name__ == "__main__":
    generate_baseline_dag()