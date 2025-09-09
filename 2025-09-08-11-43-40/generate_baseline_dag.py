#!/usr/bin/env python3
"""
Generate baseline MoE deployment DAG with:
- 16 GPUs total
- TP=8, PP=2
- 4 experts per GPU
- 4 layers total
"""

import graphviz

def create_baseline_dag():
    dot = graphviz.Digraph('baseline_moe_deployment', 
                          comment='Baseline MoE Deployment DAG',
                          format='svg')
    
    # Set graph attributes
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input Tokens\n[1024, 10000, 8192]', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Stage 0: Layers 0-1 on GPUs 0-7
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Stage 0 (Layers 0-1)\nGPUs 0-7', style='rounded')
        
        # Layer 0
        layer0_id = 0
        for gpu_id in range(8):
            gpu_name = f'gpu{gpu_id}'
            
            # Attention layer - shared across TP group
            attn_name = f'layer{layer0_id}_attn_gpu{gpu_id}'
            stage0.node(attn_name, f'Attention\n[1024, 10000, 8192]\nTP shard {gpu_id%8}\nGPU {gpu_id}', 
                       fillcolor='lightcoral')
            
            # MoE layer - 4 experts per GPU
            moe_name = f'layer{layer0_id}_moe_gpu{gpu_id}'
            stage0.node(moe_name, f'MoE Layer {layer0_id}\n[1024, 10000, 8192]\nExperts {gpu_id*4}-{(gpu_id*4)+3}\nGPU {gpu_id}', 
                       fillcolor='lightyellow')
            
            # Expert MLPs
            for expert_id in range(4):
                expert_name = f'layer{layer0_id}_expert{gpu_id*4+expert_id}_gpu{gpu_id}'
                stage0.node(expert_name, 
                           f'Expert {gpu_id*4+expert_id}\n[1024, 10000, 8192]→[1024, 10000, 32768]→[1024, 10000, 8192]\nGPU {gpu_id}',
                           fillcolor='lightsteelblue')
    
    # Stage 1: Layers 2-3 on GPUs 8-15
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Stage 1 (Layers 2-3)\nGPUs 8-15', style='rounded')
        
        # Layer 2
        layer2_id = 2
        for gpu_id in range(8, 16):
            actual_gpu = gpu_id
            
            # Attention layer
            attn_name = f'layer{layer2_id}_attn_gpu{actual_gpu}'
            stage1.node(attn_name, f'Attention\n[1024, 10000, 8192]\nTP shard {(actual_gpu-8)%8}\nGPU {actual_gpu}', 
                       fillcolor='lightcoral')
            
            # MoE layer
            moe_name = f'layer{layer2_id}_moe_gpu{actual_gpu}'
            stage1.node(moe_name, f'MoE Layer {layer2_id}\n[1024, 10000, 8192]\nExperts {(actual_gpu-8)*4}-{(actual_gpu-8)*4+3}\nGPU {actual_gpu}', 
                       fillcolor='lightyellow')
            
            # Expert MLPs
            for expert_id in range(4):
                expert_name = f'layer{layer2_id}_expert{(actual_gpu-8)*4+expert_id}_gpu{actual_gpu}'
                stage1.node(expert_name, 
                           f'Expert {(actual_gpu-8)*4+expert_id}\n[1024, 10000, 8192]→[1024, 10000, 32768]→[1024, 10000, 8192]\nGPU {actual_gpu}',
                           fillcolor='lightsteelblue')
    
    # Add remaining layers (1 and 3)
    # Layer 1 on GPUs 0-7
    layer1_id = 1
    for gpu_id in range(8):
        attn_name = f'layer{layer1_id}_attn_gpu{gpu_id}'
        dot.node(attn_name, f'Attention\n[1024, 10000, 8192]\nTP shard {gpu_id%8}\nGPU {gpu_id}', 
                fillcolor='lightcoral')
        
        moe_name = f'layer{layer1_id}_moe_gpu{gpu_id}'
        dot.node(moe_name, f'MoE Layer {layer1_id}\n[1024, 10000, 8192]\nExperts {gpu_id*4+16}-{(gpu_id*4)+19}\nGPU {gpu_id}', 
                fillcolor='lightyellow')
        
        for expert_id in range(4):
            expert_name = f'layer{layer1_id}_expert{gpu_id*4+16+expert_id}_gpu{gpu_id}'
            dot.node(expert_name, 
                    f'Expert {gpu_id*4+16+expert_id}\n[1024, 10000, 8192]→[1024, 10000, 32768]→[1024, 10000, 8192]\nGPU {gpu_id}',
                    fillcolor='lightsteelblue')
    
    # Layer 3 on GPUs 8-15
    layer3_id = 3
    for gpu_id in range(8, 16):
        actual_gpu = gpu_id
        attn_name = f'layer{layer3_id}_attn_gpu{actual_gpu}'
        dot.node(attn_name, f'Attention\n[1024, 10000, 8192]\nTP shard {(actual_gpu-8)%8}\nGPU {actual_gpu}', 
                fillcolor='lightcoral')
        
        moe_name = f'layer{layer3_id}_moe_gpu{actual_gpu}'
        dot.node(moe_name, f'MoE Layer {layer3_id}\n[1024, 10000, 8192]\nExperts {(actual_gpu-8)*4+48}-{(actual_gpu-8)*4+51}\nGPU {actual_gpu}', 
                fillcolor='lightyellow')
        
        for expert_id in range(4):
            expert_name = f'layer{layer3_id}_expert{(actual_gpu-8)*4+48+expert_id}_gpu{actual_gpu}'
            dot.node(expert_name, 
                    f'Expert {(actual_gpu-8)*4+48+expert_id}\n[1024, 10000, 8192]→[1024, 10000, 32768]→[1024, 10000, 8192]\nGPU {actual_gpu}',
                    fillcolor='lightsteelblue')
    
    # Communication nodes
    dot.attr('node', shape='parallelogram', fillcolor='lightpink')
    
    # TP All-reduce nodes
    for layer in range(4):
        for tp_group in range(2):
            tp_node = f'tp_allreduce_layer{layer}_group{tp_group}'
            start_gpu = tp_group * 8
            dot.node(tp_node, f'TP All-Reduce\nLayer {layer}\nGPUs {start_gpu}-{start_gpu+7}', 
                    fillcolor='lightpink')
    
    # Pipeline communication nodes
    pp_send_01 = 'pp_send_stage0_to_stage1'
    pp_recv_10 = 'pp_recv_stage0_to_stage1'
    dot.node(pp_send_01, 'Pipeline Send\nStage 0 → Stage 1\n[1024, 10000, 8192]', 
            fillcolor='lightgreen')
    dot.node(pp_recv_10, 'Pipeline Recv\nStage 0 → Stage 1\n[1024, 10000, 8192]', 
            fillcolor='lightgreen')
    
    # Gating nodes
    dot.attr('node', shape='diamond', fillcolor='orange')
    for layer in range(4):
        gate_name = f'gating_layer{layer}'
        dot.node(gate_name, f'Top-K Gating\nLayer {layer}\nSelect 2 experts\nfrom 16 total', 
                fillcolor='orange')
    
    # Output node
    dot.node('output', 'Output Tokens\n[1024, 10000, 8192]', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Connect the flow
    # Input to Layer 0
    dot.edge('input', 'gating_layer0')
    
    # Layer 0 connections
    for gpu_id in range(8):
        dot.edge('gating_layer0', f'layer0_attn_gpu{gpu_id}')
        dot.edge(f'layer0_attn_gpu{gpu_id}', f'layer0_moe_gpu{gpu_id}')
        
        # Connect to experts
        for expert_id in range(4):
            dot.edge(f'layer0_moe_gpu{gpu_id}', 
                    f'layer0_expert{gpu_id*4+expert_id}_gpu{gpu_id}')
            # Expert outputs back to MoE layer
            dot.edge(f'layer0_expert{gpu_id*4+expert_id}_gpu{gpu_id}', 
                    f'layer0_moe_gpu{gpu_id}')
    
    # TP all-reduce for layer 0
    for gpu_id in range(8):
        dot.edge(f'layer0_moe_gpu{gpu_id}', 'tp_allreduce_layer0_group0')
    
    # Layer 1 connections
    dot.edge('tp_allreduce_layer0_group0', 'gating_layer1')
    for gpu_id in range(8):
        dot.edge('gating_layer1', f'layer1_attn_gpu{gpu_id}')
        dot.edge(f'layer1_attn_gpu{gpu_id}', f'layer1_moe_gpu{gpu_id}')
        
        for expert_id in range(4):
            dot.edge(f'layer1_moe_gpu{gpu_id}', 
                    f'layer1_expert{gpu_id*4+16+expert_id}_gpu{gpu_id}')
            dot.edge(f'layer1_expert{gpu_id*4+16+expert_id}_gpu{gpu_id}', 
                    f'layer1_moe_gpu{gpu_id}')
    
    # TP all-reduce for layer 1
    for gpu_id in range(8):
        dot.edge(f'layer1_moe_gpu{gpu_id}', 'tp_allreduce_layer1_group0')
    
    # Pipeline communication between stages
    dot.edge('tp_allreduce_layer1_group0', pp_send_01)
    dot.edge(pp_send_01, pp_recv_10)
    dot.edge(pp_recv_10, 'gating_layer2')
    
    # Layer 2 connections
    for gpu_id in range(8, 16):
        dot.edge('gating_layer2', f'layer2_attn_gpu{gpu_id}')
        dot.edge(f'layer2_attn_gpu{gpu_id}', f'layer2_moe_gpu{gpu_id}')
        
        for expert_id in range(4):
            dot.edge(f'layer2_moe_gpu{gpu_id}', 
                    f'layer2_expert{(gpu_id-8)*4+expert_id}_gpu{gpu_id}')
            dot.edge(f'layer2_expert{(gpu_id-8)*4+expert_id}_gpu{gpu_id}', 
                    f'layer2_moe_gpu{gpu_id}')
    
    # TP all-reduce for layer 2
    for gpu_id in range(8, 16):
        dot.edge(f'layer2_moe_gpu{gpu_id}', 'tp_allreduce_layer2_group1')
    
    # Layer 3 connections
    dot.edge('tp_allreduce_layer2_group1', 'gating_layer3')
    for gpu_id in range(8, 16):
        dot.edge('gating_layer3', f'layer3_attn_gpu{gpu_id}')
        dot.edge(f'layer3_attn_gpu{gpu_id}', f'layer3_moe_gpu{gpu_id}')
        
        for expert_id in range(4):
            dot.edge(f'layer3_moe_gpu{gpu_id}', 
                    f'layer3_expert{(gpu_id-8)*4+48+expert_id}_gpu{gpu_id}')
            dot.edge(f'layer3_expert{(gpu_id-8)*4+48+expert_id}_gpu{gpu_id}', 
                    f'layer3_moe_gpu{gpu_id}')
    
    # TP all-reduce for layer 3
    for gpu_id in range(8, 16):
        dot.edge(f'layer3_moe_gpu{gpu_id}', 'tp_allreduce_layer3_group1')
    
    # Final output
    dot.edge('tp_allreduce_layer3_group1', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/2025-09-08-11-43-40/baseline_moe_deployment', 
               format='svg', cleanup=True)
    
    # Also save as .dot file
    with open('/home/wzc/data/file-share/2025-09-08-11-43-40/baseline_moe_deployment.dot', 'w') as f:
        f.write(dag.source)
    
    print("Baseline DAG generated successfully")