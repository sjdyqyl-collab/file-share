import graphviz
from typing import Dict, List, Tuple

def create_baseline_dag():
    """Create baseline DAG with 16 GPUs, 4 experts per GPU"""
    dot = graphviz.Digraph('baseline_moe_dag', comment='Baseline MoE Deployment (16 GPUs, 4 experts/GPU)')
    dot.attr(rankdir='TB', size='20,20')
    
    # Define colors for different GPU groups
    colors = {
        'gpu0-7': 'lightblue',    # Pipeline stage 0
        'gpu8-15': 'lightgreen'   # Pipeline stage 1
    }
    
    # Input processing
    dot.node('input', 'Model Input\n[1024, seq_len, 4096]', shape='ellipse', style='filled', fillcolor='yellow')
    
    # Process 4 layers
    for layer in range(4):
        layer_prefix = f'layer{layer}'
        
        # MHA across all GPUs (tensor parallel)
        for gpu in range(16):
            gpu_id = gpu
            stage = 0 if gpu < 8 else 1
            color = colors['gpu0-7'] if gpu < 8 else colors['gpu8-15']
            
            # MHA computation
            mha_node = f'{layer_prefix}_mha_gpu{gpu}'
            dot.node(mha_node, f'MHA TP-Shard\nGPU{gpu}\n[1024, seq_len, 512]\n→ [1024, seq_len, 512]', 
                    shape='rectangle', style='filled', fillcolor=color)
            
            # MHA All-reduce
            mha_ar = f'{layer_prefix}_mha_ar_gpu{gpu}'
            dot.node(mha_ar, f'MHA All-Reduce\nGPU{gpu}\n[1024, seq_len, 4096]', 
                    shape='ellipse', style='filled', fillcolor='lightyellow')
            
            # Residual add
            residual1 = f'{layer_prefix}_residual1_gpu{gpu}'
            dot.node(residual1, f'Residual Add\nGPU{gpu}\n[1024, seq_len, 4096]', 
                    shape='parallelogram', style='filled', fillcolor=color)
            
            # Layer norm
            ln1 = f'{layer_prefix}_ln1_gpu{gpu}'
            dot.node(ln1, f'LayerNorm\nGPU{gpu}\n[1024, seq_len, 4096]', 
                    shape='rectangle', style='filled', fillcolor=color)
            
            # Gating network
            gate = f'{layer_prefix}_gate_gpu{gpu}'
            dot.node(gate, f'Gating Network\nGPU{gpu}\n[1024, seq_len, 4096]\n→ [1024, seq_len, 16]', 
                    shape='diamond', style='filled', fillcolor=color)
            
            # 4 Experts per GPU (experts are distributed)
            expert_base = (gpu // 4) * 4 + (gpu % 4) * 4
            for exp_idx in range(4):
                expert_id = expert_base + exp_idx
                expert = f'{layer_prefix}_expert{expert_id}_gpu{gpu}'
                dot.node(expert, f'Expert {expert_id}\nGPU{gpu}\n[1024, seq_len, 4096]\n→ [1024, seq_len, 4096]', 
                        shape='rectangle', style='filled', fillcolor=color)
                
                # Expert selection (dashed line from gate)
                dot.edge(gate, expert, style='dashed', label=f'expert {expert_id}')
            
            # Expert aggregation
            expert_agg = f'{layer_prefix}_expert_agg_gpu{gpu}'
            dot.node(expert_agg, f'Expert Aggregation\nGPU{gpu}\n[1024, seq_len, 4096]', 
                    shape='parallelogram', style='filled', fillcolor=color)
            
            # Second residual add
            residual2 = f'{layer_prefix}_residual2_gpu{gpu}'
            dot.node(residual2, f'Residual Add\nGPU{gpu}\n[1024, seq_len, 4096]', 
                    shape='parallelogram', style='filled', fillcolor=color)
            
            # Layer norm 2
            ln2 = f'{layer_prefix}_ln2_gpu{gpu}'
            dot.node(ln2, f'LayerNorm\nGPU{gpu}\n[1024, seq_len, 4096]', 
                    shape='rectangle', style='filled', fillcolor=color)
            
            # Pipeline communication between stages
            if layer < 3 and gpu < 8:
                next_stage = f'layer{layer+1}_mha_gpu{gpu+8}'
                dot.edge(ln2, next_stage, label='pipeline\nstage transfer', style='dotted')
            elif layer < 3 and gpu >= 8:
                next_stage = f'layer{layer+1}_mha_gpu{gpu-8}'
                dot.edge(ln2, next_stage, label='pipeline\nstage transfer', style='dotted')
    
    # Output
    dot.node('output', 'Model Output\n[1024, seq_len, 4096]', shape='ellipse', style='filled', fillcolor='yellow')
    
    # Connect input to first layer
    for gpu in range(16):
        dot.edge('input', f'layer0_mha_gpu{gpu}')
    
    # Connect last layer to output
    for gpu in range(16):
        dot.edge(f'layer3_ln2_gpu{gpu}', 'output')
    
    return dot

def create_proposed_dag():
    """Create proposed DAG with 64 GPUs, 1 expert per GPU"""
    dot = graphviz.Digraph('proposed_moe_dag', comment='Proposed MoE Deployment (64 GPUs, 1 expert/GPU)')
    dot.attr(rankdir='TB', size='30,30')
    
    # Define node colors for different nodes
    node_colors = {
        'mha': 'lightblue',
        'gate': 'lightgreen',
        'expert': 'lightcoral',
        'comm': 'lightyellow',
        'agg': 'lightpink'
    }
    
    # Input processing
    dot.node('input', 'Model Input\n[1024, seq_len, 4096]', shape='ellipse', style='filled', fillcolor='yellow')
    
    # Process 4 layers
    for layer in range(4):
        layer_prefix = f'layer{layer}'
        
        # MHA - replicated on all GPUs (no tensor parallelism)
        for gpu in range(64):
            # MHA computation (full model on each GPU)
            mha_node = f'{layer_prefix}_mha_gpu{gpu}'
            dot.node(mha_node, f'MHA\nGPU{gpu}\n[1024, seq_len, 4096]\n→ [1024, seq_len, 4096]', 
                    shape='rectangle', style='filled', fillcolor=node_colors['mha'])
            
            # Residual add
            residual1 = f'{layer_prefix}_residual1_gpu{gpu}'
            dot.node(residual1, f'Residual Add\nGPU{gpu}\n[1024, seq_len, 4096]', 
                    shape='parallelogram', style='filled', fillcolor=node_colors['mha'])
            
            # Layer norm
            ln1 = f'{layer_prefix}_ln1_gpu{gpu}'
            dot.node(ln1, f'LayerNorm\nGPU{gpu}\n[1024, seq_len, 4096]', 
                    shape='rectangle', style='filled', fillcolor=node_colors['mha'])
            
            # Gating network (on each GPU for routing decisions)
            gate = f'{layer_prefix}_gate_gpu{gpu}'
            dot.node(gate, f'Gating Network\nGPU{gpu}\n[1024, seq_len, 4096]\n→ [1024, seq_len, 16]', 
                    shape='diamond', style='filled', fillcolor=node_colors['gate'])
            
            # Expert assignment - one expert per GPU
            expert_id = layer * 16 + (gpu % 16)  # 16 experts per layer across 64 GPUs
            expert = f'{layer_prefix}_expert{expert_id}_gpu{gpu}'
            dot.node(expert, f'Expert {expert_id}\nGPU{gpu}\n[batch, seq_len, 4096]\n→ [batch, seq_len, 4096]', 
                    shape='rectangle', style='filled', fillcolor=node_colors['expert'])
            
            # Token routing communication (async)
            route_comm = f'{layer_prefix}_route_gpu{gpu}'
            dot.node(route_comm, f'Token Routing\nGPU{gpu}\nAsync Send/Recv', 
                    shape='ellipse', style='filled', fillcolor=node_colors['comm'])
            
            # Expert computation
            expert_compute = f'{layer_prefix}_expert_compute_gpu{gpu}'
            dot.node(expert_compute, f'Expert {expert_id} Compute\nGPU{gpu}\n[batch, seq_len, 4096]', 
                    shape='rectangle', style='filled', fillcolor=node_colors['expert'])
            
            # Token aggregation (async gather)
            token_gather = f'{layer_prefix}_gather_gpu{gpu}'
            dot.node(token_gather, f'Token Gather\nGPU{gpu}\nAsync Gather', 
                    shape='ellipse', style='filled', fillcolor=node_colors['comm'])
            
            # Second residual add
            residual2 = f'{layer_prefix}_residual2_gpu{gpu}'
            dot.node(residual2, f'Residual Add\nGPU{gpu}\n[1024, seq_len, 4096]', 
                    shape='parallelogram', style='filled', fillcolor=node_colors['agg'])
            
            # Layer norm 2
            ln2 = f'{layer_prefix}_ln2_gpu{gpu}'
            dot.node(ln2, f'LayerNorm\nGPU{gpu}\n[1024, seq_len, 4096]', 
                    shape='rectangle', style='filled', fillcolor=node_colors['agg'])
            
            # Connect nodes for this GPU
            dot.edge('input', mha_node)
            dot.edge(mha_node, residual1)
            dot.edge(residual1, ln1)
            dot.edge(ln1, gate)
            dot.edge(gate, route_comm, style='dashed', label='routing decision')
            dot.edge(route_comm, expert_compute)
            dot.edge(expert_compute, token_gather)
            dot.edge(token_gather, residual2)
            dot.edge(residual2, ln2)
            
            # Cross-GPU communication for token routing
            if gpu < 63:
                dot.edge(route_comm, f'{layer_prefix}_expert_compute_gpu{gpu+1}', 
                        label='token send', style='dotted')
                dot.edge(token_gather, f'{layer_prefix}_gather_gpu{gpu+1}', 
                        label='token recv', style='dotted')
    
    # Output
    dot.node('output', 'Model Output\n[1024, seq_len, 4096]', shape='ellipse', style='filled', fillcolor='yellow')
    
    # Connect last layer to output
    for gpu in range(64):
        dot.edge(f'layer3_ln2_gpu{gpu}', 'output')
    
    return dot

if __name__ == '__main__':
    # Generate baseline DAG
    baseline_dag = create_baseline_dag()
    baseline_dag.render('/home/wzc/data/file-share/submission/baseline_moe_dag', format='svg', cleanup=True)
    
    # Generate proposed DAG
    proposed_dag = create_proposed_dag()
    proposed_dag.render('/home/wzc/data/file-share/submission/proposed_moe_dag', format='svg', cleanup=True)
    
    print("DAGs generated successfully!")
    print("- Baseline: /home/wzc/data/file-share/submission/baseline_moe_dag.svg")
    print("- Proposed: /home/wzc/data/file-share/submission/proposed_moe_dag.svg")