#!/usr/bin/env python3

import graphviz

def generate_proposed_dag():
    """Generate DAG for Large-Scale Cross-Node Expert Parallelism (Proposed)"""
    
    dot = graphviz.Digraph('proposed_moe_parallelism', format='svg')
    dot.attr(rankdir='TB', size='60,40', ranksep='2.5', nodesep='1.5')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    dot.attr('node', shape='diamond', style='filled', fillcolor='lightcoral')  # Communication
    
    # Global input
    dot.node('input', 'Input\n[1024 seqs, 10000 tokens, 8192 dim]', shape='ellipse', fillcolor='lightblue')
    
    # Model has 4 layers, each with 16 experts
    # 64 GPUs total, each GPU has 1 expert (64 experts / 64 GPUs = 1 expert per GPU)
    # EP=64, PP=4, TP=1
    
    # Pipeline stages: 4 layers, each layer is a pipeline stage
    # Layer 0: GPUs 0-15 (Node 0-1)
    # Layer 1: GPUs 16-31 (Node 2-3)
    # Layer 2: GPUs 32-47 (Node 4-5)
    # Layer 3: GPUs 48-63 (Node 6-7)
    
    prev_output = 'input'
    
    for layer in range(4):
        # Determine GPU range for this layer
        gpu_start = layer * 16
        gpu_end = gpu_start + 15
        node_start = layer * 2
        node_end = node_start + 1
        
        # Layer norm before attention (replicated across all GPUs for this layer)
        ln1_name = f'layer{layer}_ln1'
        dot.node(ln1_name, f'LayerNorm\n[1024, 10000, 8192]\nAll GPUs {gpu_start}-{gpu_end}', 
                shape='rectangle', fillcolor='lightgreen')
        dot.edge(prev_output, ln1_name)
        
        # Multi-head attention (no tensor parallelism, replicated computation)
        attn_name = f'layer{layer}_attn'
        dot.node(attn_name, f'Multi-Head Attention\n16 heads × 512 dim\n[1024, 10000, 8192]\nAll GPUs {gpu_start}-{gpu_end}', 
                shape='rectangle', fillcolor='lightgreen')
        dot.edge(ln1_name, attn_name)
        
        # Residual connection after attention
        residual1_name = f'layer{layer}_residual1'
        dot.node(residual1_name, f'Residual Add\n[1024, 10000, 8192]\nAll GPUs {gpu_start}-{gpu_end}', 
                shape='parallelogram', fillcolor='lightyellow')
        dot.edge(prev_output, residual1_name)
        dot.edge(attn_name, residual1_name)
        
        # Layer norm before MoE
        ln2_name = f'layer{layer}_ln2'
        dot.node(ln2_name, f'LayerNorm\n[1024, 10000, 8192]\nAll GPUs {gpu_start}-{gpu_end}', 
                shape='rectangle', fillcolor='lightgreen')
        dot.edge(residual1_name, ln2_name)
        
        # Gate for expert selection (distributed across all GPUs)
        gate_name = f'layer{layer}_gate'
        dot.node(gate_name, f'Gate\nSelect top-2 experts\n[1024, 10000, 16]\nAll GPUs {gpu_start}-{gpu_end}', 
                shape='parallelogram', fillcolor='lightyellow')
        dot.edge(ln2_name, gate_name)
        
        # Expert computation - 1 expert per GPU
        expert_outputs = []
        for expert_idx in range(16):
            expert_id = layer * 16 + expert_idx
            gpu_id = expert_id
            node_id = gpu_id // 8
            
            expert_name = f'layer{layer}_expert{expert_id}_gpu{gpu_id}'
            dot.node(expert_name, f'Expert {expert_id}\nMLP\n[32768 hidden]\nGPU {gpu_id} (Node {node_id})', 
                    shape='rectangle', fillcolor='lightgreen')
            
            # Token routing from gate to expert (asynchronous)
            route_name = f'route_layer{layer}_expert{expert_id}'
            dot.node(route_name, f'Async Token Routing\nto Expert {expert_id}\nGPU {gpu_id} (Node {node_id})', 
                    shape='diamond', fillcolor='lightcoral')
            dot.edge(gate_name, route_name, style='dashed')
            dot.edge(route_name, expert_name)
            
            # Expert computation
            dot.edge(ln2_name, expert_name, style='dashed', label=f'expert {expert_id}')
            
            # Expert output routing back
            expert_return_name = f'expert_return_layer{layer}_expert{expert_id}'
            dot.node(expert_return_name, f'Async Expert Return\nfrom Expert {expert_id}\nGPU {gpu_id} (Node {node_id})', 
                    shape='diamond', fillcolor='lightcoral')
            dot.edge(expert_name, expert_return_name)
            
            expert_outputs.append(expert_return_name)
        
        # Expert output aggregation across all experts
        expert_agg_name = f'layer{layer}_expert_agg'
        dot.node(expert_agg_name, f'Aggregate Expert Outputs\n[1024, 10000, 8192]\nAll GPUs {gpu_start}-{gpu_end}', 
                shape='parallelogram', fillcolor='lightyellow')
        
        for expert_output in expert_outputs:
            dot.edge(expert_output, expert_agg_name)
        
        # Residual connection after MoE
        residual2_name = f'layer{layer}_residual2'
        dot.node(residual2_name, f'Residual Add\n[1024, 10000, 8192]\nAll GPUs {gpu_start}-{gpu_end}', 
                shape='parallelogram', fillcolor='lightyellow')
        dot.edge(residual1_name, residual2_name)
        dot.edge(expert_agg_name, residual2_name)
        
        # Pipeline communication between layers
        if layer < 3:
            next_gpu_start = (layer + 1) * 16
            next_gpu_end = next_gpu_start + 15
            comm_name = f'pipeline_comm_layer{layer}'
            dot.node(comm_name, f'Pipeline Communication\nLayer {layer} → Layer {layer+1}\n[1024, 10000, 8192]\nGPU {gpu_end} → GPU {next_gpu_start}', 
                    shape='diamond', fillcolor='lightcoral')
            dot.edge(residual2_name, comm_name)
            prev_output = comm_name
        else:
            prev_output = residual2_name
    
    # Global output
    dot.node('output', 'Output\n[1024 seqs, 10000 tokens, 8192 dim]', shape='ellipse', fillcolor='lightblue')
    dot.edge(prev_output, 'output')
    
    # Save files
    dot.render('/home/wzc/data/file-share/2025-09-05-17-32-21/proposed_moe_parallelism', format='svg', cleanup=False)
    
    # Also save as .dot file
    with open('/home/wzc/data/file-share/2025-09-05-17-32-21/proposed_moe_parallelism.dot', 'w') as f:
        f.write(dot.source)
    
    return dot.source

if __name__ == "__main__":
    generate_proposed_dag()