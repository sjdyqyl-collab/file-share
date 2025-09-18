#!/usr/bin/env python3

import graphviz
import json

def create_baseline_dag():
    """Baseline DAG: 16 GPUs, TP=8, PP=2, 4 experts/GPU"""
    dot = graphviz.Digraph('baseline_moe', format='svg')
    dot.attr(rankdir='TB', size='30,20')
    
    # Input
    dot.node('input', 'Input\\n[1024×10000×8192]', 
             shape='parallelogram', fillcolor='lightgreen', style='filled')
    
    # 4 layers
    for layer_id in range(4):
        stage = layer_id % 2
        gpu_base = 0 if stage == 0 else 8
        
        # Layer cluster
        with dot.subgraph(name=f'layer_{layer_id}') as layer:
            layer.attr(label=f'Layer {layer_id} (Stage {stage})')
            
            # Gating
            gate = f'gate_{layer_id}'
            layer.node(gate, f'Gating\\nLayer {layer_id}', 
                      shape='diamond', fillcolor='yellow', style='filled')
            
            # Experts (4 per GPU)
            for exp_id in range(16):
                gpu_id = gpu_base + (exp_id // 4)
                exp = f'expert_{layer_id}_{exp_id}'
                layer.node(exp, f'Expert {exp_id}\\n[8192→32768→8192]\\nGPU {gpu_id}', 
                          shape='rectangle', fillcolor='lightcoral', style='filled')
                layer.edge(gate, exp, style='dashed')
            
            # Aggregation
            agg = f'agg_{layer_id}'
            layer.node(agg, f'Aggregate\\nLayer {layer_id}', 
                      shape='parallelogram', fillcolor='purple', style='filled')
            
            for exp_id in range(16):
                layer.edge(f'expert_{layer_id}_{exp_id}', agg)
    
    # Connect layers
    for i in range(3):
        dot.edge(f'agg_{i}', f'gate_{i+1}')
    dot.edge('input', 'gate_0')
    dot.edge('agg_3', 'output')
    
    dot.node('output', 'Output\\n[1024×10000×8192]', 
             shape='parallelogram', fillcolor='lightgreen', style='filled')
    
    return dot

def create_proposed_dag():
    """Proposed DAG: 64 GPUs, EP=16, 1 expert/GPU"""
    dot = graphviz.Digraph('proposed_moe', format='svg')
    dot.attr(rankdir='TB', size='40,30')
    
    # Input
    dot.node('input', 'Input\\n[1024×10000×8192]', 
             shape='parallelogram', fillcolor='lightgreen', style='filled')
    
    # 4 layers
    for layer_id in range(4):
        gpu_base = layer_id * 16
        
        # Layer cluster
        with dot.subgraph(name=f'layer_{layer_id}') as layer:
            layer.attr(label=f'Layer {layer_id} (GPUs {gpu_base}-{gpu_base+15})')
            
            # Gating
            gate = f'gate_{layer_id}'
            layer.node(gate, f'Gating\\nLayer {layer_id}\\nGPU {gpu_base}', 
                      shape='diamond', fillcolor='yellow', style='filled')
            
            # Router
            router = f'router_{layer_id}'
            layer.node(router, 'Token Router\\n[Cross-node]', 
                      shape='parallelogram', fillcolor='orange', style='filled')
            layer.edge(gate, router)
            
            # Experts (1 per GPU)
            for exp_id in range(16):
                gpu_id = gpu_base + exp_id
                exp = f'expert_{layer_id}_{exp_id}'
                layer.node(exp, f'Expert {exp_id}\\n[8192→32768→8192]\\nGPU {gpu_id}', 
                          shape='rectangle', fillcolor='lightcoral', style='filled')
                layer.edge(router, exp, style='dashed', label=f'GPU {gpu_id}')
            
            # Aggregator
            agg = f'agg_{layer_id}'
            layer.node(agg, f'Aggregate\\nLayer {layer_id}', 
                      shape='parallelogram', fillcolor='purple', style='filled')
            
            for exp_id in range(16):
                layer.edge(f'expert_{layer_id}_{exp_id}', agg)
    
    # Connect layers
    for i in range(3):
        dot.edge(f'agg_{i}', f'gate_{i+1}')
    dot.edge('input', 'gate_0')
    dot.edge('agg_3', 'output')
    
    dot.node('output', 'Output\\n[1024×10000×8192]', 
             shape='parallelogram', fillcolor='lightgreen', style='filled')
    
    return dot

# Generate all DAGs
print("Generating DAGs...")

# Baseline
baseline = create_baseline_dag()
baseline.render('/home/wzc/data/file-share/2025-09-08-16-15-08/baseline_final', format='svg')
with open('/home/wzc/data/file-share/2025-09-08-16-15-08/baseline_final.dot', 'w') as f:
    f.write(baseline.source)

# Proposed
proposed = create_proposed_dag()
proposed.render('/home/wzc/data/file-share/2025-09-08-16-15-08/proposed_final', format='svg')
with open('/home/wzc/data/file-share/2025-09-08-16-15-08/proposed_final.dot', 'w') as f:
    f.write(proposed.source)

# Create summary
summary = {
    "generated_dags": [
        {
            "name": "Baseline MoE Deployment",
            "description": "16 GPUs with TP=8, PP=2, 4 experts per GPU",
            "configuration": {
                "total_gpus": 16,
                "experts_per_gpu": 4,
                "total_experts": 64
            },
            "files": {
                "svg": "/home/wzc/data/file-share/2025-09-08-16-15-08/baseline_final.svg",
                "dot": "/home/wzc/data/file-share/2025-09-08-16-15-08/baseline_final.dot"
            }
        },
        {
            "name": "Proposed Large-Scale Expert Parallelism",
            "description": "64 GPUs with EP=16, 1 expert per GPU",
            "configuration": {
                "total_gpus": 64,
                "experts_per_gpu": 1,
                "total_experts": 64
            },
            "files": {
                "svg": "/home/wzc/data/file-share/2025-09-08-16-15-08/proposed_final.svg",
                "dot": "/home/wzc/data/file-share/2025-09-08-16-15-08/proposed_final.dot"
            }
        }
    ]
}

with open('/home/wzc/data/file-share/2025-09-08-16-15-08/final_dags.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("All DAGs generated successfully!")