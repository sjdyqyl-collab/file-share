#!/usr/bin/env python3
import os
from graphviz import Digraph

def create_baseline_dag():
    """Create baseline DAG with TP=8, PP=2, 8 experts per GPU"""
    dot = Digraph('baseline_moe_dag', comment='Baseline MoE Deployment (TP=8, PP=2)')
    dot.attr(rankdir='TB', size='20,30')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    dot.attr('node', shape='diamond', style='filled', fillcolor='lightcoral')  # Communication
    
    # Global input
    dot.node('input', 'Global Input\\nInput: [batch_size=1024, seq_len=10000, hidden=8192]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Pipeline stage 0 (layers 0-1) - GPUs 0-7
    with dot.subgraph(name='cluster_pipeline_stage_0') as c0:
        c0.attr(label='Pipeline Stage 0 (Layers 0-1)\\nGPUs 0-7', style='dashed')
        
        # Layer 0
        with c0.subgraph(name='cluster_layer_0') as layer0:
            layer0.attr(label='Layer 0', style='dotted')
            
            # Attention block for layer 0
            with layer0.subgraph(name='cluster_layer0_attention') as attn0:
                attn0.attr(label='Multi-Head Attention', style='rounded')
                
                # QKV projection (TP=8)
                for i in range(8):
                    dot.node(f'layer0_qkv_gpu{i}', f'QKV Projection\\nGPU: {i}\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,16,512]', 
                             shape='rectangle', fillcolor='lightgreen')
                    
                # Attention computation (TP=8)
                for i in range(8):
                    dot.node(f'layer0_attn_gpu{i}', f'Attention\\nGPU: {i}\\nInput: [1024,10000,16,512]\\nOutput: [1024,10000,16,512]', 
                             shape='rectangle', fillcolor='lightgreen')
                    
                # Output projection (TP=8)
                for i in range(8):
                    dot.node(f'layer0_out_gpu{i}', f'Output Projection\\nGPU: {i}\\nInput: [1024,10000,16,512]\\nOutput: [1024,10000,1024]', 
                             shape='rectangle', fillcolor='lightgreen')
                
                # All-reduce for attention output
                dot.node('layer0_attn_allreduce', 'All-Reduce\\nTP Group [0-7]\\nInput: [1024,10000,1024]\\nOutput: [1024,10000,8192]', 
                         shape='diamond', fillcolor='lightcoral')
                
                # Residual add
                dot.node('layer0_residual', 'Residual Add\\nInput: [1024,10000,8192], [1024,10000,8192]\\nOutput: [1024,10000,8192]', 
                         shape='parallelogram', fillcolor='lightyellow')
                
                # Layer norm
                dot.node('layer0_layernorm', 'Layer Norm\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]', 
                         shape='rectangle', fillcolor='lightgreen')
            
            # MoE block for layer 0
            with layer0.subgraph(name='cluster_layer0_moe') as moe0:
                moe0.attr(label='MoE Layer (16 Experts)', style='rounded')
                
                # Gate computation
                dot.node('layer0_gate', 'Gate\\nGPU: 0-7\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,16]', 
                         shape='parallelogram', fillcolor='lightyellow')
                
                # Expert routing (dashed lines will show connections)
                dot.node('layer0_router', 'Router\\nDistribute tokens\\nInput: [1024,10000,8192]\\nOutput: [varies per expert]', 
                         shape='parallelogram', fillcolor='lightyellow')
                
                # Experts (8 per GPU)
                for gpu in range(8):
                    for expert in range(8):
                        expert_id = gpu * 2 + (expert % 2) * 8 + expert // 2
                        dot.node(f'layer0_expert{expert_id}_gpu{gpu}', 
                                 f'Expert {expert_id}\\nGPU: {gpu}\\nInput: [batch, seq, 8192]\\nOutput: [batch, seq, 8192]', 
                                 shape='rectangle', fillcolor='lightgreen')
                
                # Expert aggregation
                dot.node('layer0_expert_agg', 'Expert Aggregation\\nGPU: 0-7\\nInput: [varies per expert]\\nOutput: [1024,10000,8192]', 
                         shape='parallelogram', fillcolor='lightyellow')
                
                # Residual add
                dot.node('layer0_moe_residual', 'Residual Add\\nInput: [1024,10000,8192], [1024,10000,8192]\\nOutput: [1024,10000,8192]', 
                         shape='parallelogram', fillcolor='lightyellow')
        
        # Layer 1 (similar structure)
        with c0.subgraph(name='cluster_layer_1') as layer1:
            layer1.attr(label='Layer 1', style='dotted')
            
            # Attention block for layer 1
            with layer1.subgraph(name='cluster_layer1_attention') as attn1:
                attn1.attr(label='Multi-Head Attention', style='rounded')
                
                for i in range(8):
                    dot.node(f'layer1_qkv_gpu{i}', f'QKV Projection\\nGPU: {i}\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,16,512]', 
                             shape='rectangle', fillcolor='lightgreen')
                    
                for i in range(8):
                    dot.node(f'layer1_attn_gpu{i}', f'Attention\\nGPU: {i}\\nInput: [1024,10000,16,512]\\nOutput: [1024,10000,16,512]', 
                             shape='rectangle', fillcolor='lightgreen')
                    
                for i in range(8):
                    dot.node(f'layer1_out_gpu{i}', f'Output Projection\\nGPU: {i}\\nInput: [1024,10000,16,512]\\nOutput: [1024,10000,1024]', 
                             shape='rectangle', fillcolor='lightgreen')
                
                dot.node('layer1_attn_allreduce', 'All-Reduce\\nTP Group [0-7]\\nInput: [1024,10000,1024]\\nOutput: [1024,10000,8192]', 
                         shape='diamond', fillcolor='lightcoral')
                
                dot.node('layer1_residual', 'Residual Add\\nInput: [1024,10000,8192], [1024,10000,8192]\\nOutput: [1024,10000,8192]', 
                         shape='parallelogram', fillcolor='lightyellow')
                
                dot.node('layer1_layernorm', 'Layer Norm\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]', 
                         shape='rectangle', fillcolor='lightgreen')
            
            # MoE block for layer 1
            with layer1.subgraph(name='cluster_layer1_moe') as moe1:
                moe1.attr(label='MoE Layer (16 Experts)', style='rounded')
                
                dot.node('layer1_gate', 'Gate\\nGPU: 0-7\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,16]', 
                         shape='parallelogram', fillcolor='lightyellow')
                
                dot.node('layer1_router', 'Router\\nDistribute tokens\\nInput: [1024,10000,8192]\\nOutput: [varies per expert]', 
                         shape='parallelogram', fillcolor='lightyellow')
                
                for gpu in range(8):
                    for expert in range(8):
                        expert_id = gpu * 2 + (expert % 2) * 8 + expert // 2
                        dot.node(f'layer1_expert{expert_id}_gpu{gpu}', 
                                 f'Expert {expert_id}\\nGPU: {gpu}\\nInput: [batch, seq, 8192]\\nOutput: [batch, seq, 8192]', 
                                 shape='rectangle', fillcolor='lightgreen')
                
                dot.node('layer1_expert_agg', 'Expert Aggregation\\nGPU: 0-7\\nInput: [varies per expert]\\nOutput: [1024,10000,8192]', 
                         shape='parallelogram', fillcolor='lightyellow')
                
                dot.node('layer1_moe_residual', 'Residual Add\\nInput: [1024,10000,8192], [1024,10000,8192]\\nOutput: [1024,10000,8192]', 
                         shape='parallelogram', fillcolor='lightyellow')
    
    # Pipeline stage 1 (layers 2-3) - GPUs 8-15
    with dot.subgraph(name='cluster_pipeline_stage_1') as c1:
        c1.attr(label='Pipeline Stage 1 (Layers 2-3)\\nGPUs 8-15', style='dashed')
        
        # Communication between stages
        dot.node('pipeline_send_0_1', 'Pipeline Send\\nStage 0 → Stage 1\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]', 
                 shape='diamond', fillcolor='lightcoral')
        
        # Layer 2
        with c1.subgraph(name='cluster_layer_2') as layer2:
            layer2.attr(label='Layer 2', style='dotted')
            
            # Attention block for layer 2
            with layer2.subgraph(name='cluster_layer2_attention') as attn2:
                attn2.attr(label='Multi-Head Attention', style='rounded')
                
                for i in range(8, 16):
                    dot.node(f'layer2_qkv_gpu{i}', f'QKV Projection\\nGPU: {i}\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,16,512]', 
                             shape='rectangle', fillcolor='lightgreen')
                    
                for i in range(8, 16):
                    dot.node(f'layer2_attn_gpu{i}', f'Attention\\nGPU: {i}\\nInput: [1024,10000,16,512]\\nOutput: [1024,10000,16,512]', 
                             shape='rectangle', fillcolor='lightgreen')
                    
                for i in range(8, 16):
                    dot.node(f'layer2_out_gpu{i}', f'Output Projection\\nGPU: {i}\\nInput: [1024,10000,16,512]\\nOutput: [1024,10000,1024]', 
                             shape='rectangle', fillcolor='lightgreen')
                
                dot.node('layer2_attn_allreduce', 'All-Reduce\\nTP Group [8-15]\\nInput: [1024,10000,1024]\\nOutput: [1024,10000,8192]', 
                         shape='diamond', fillcolor='lightcoral')
                
                dot.node('layer2_residual', 'Residual Add\\nInput: [1024,10000,8192], [1024,10000,8192]\\nOutput: [1024,10000,8192]', 
                         shape='parallelogram', fillcolor='lightyellow')
                
                dot.node('layer2_layernorm', 'Layer Norm\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]', 
                         shape='rectangle', fillcolor='lightgreen')
            
            # MoE block for layer 2
            with layer2.subgraph(name='cluster_layer2_moe') as moe2:
                moe2.attr(label='MoE Layer (16 Experts)', style='rounded')
                
                dot.node('layer2_gate', 'Gate\\nGPU: 8-15\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,16]', 
                         shape='parallelogram', fillcolor='lightyellow')
                
                dot.node('layer2_router', 'Router\\nDistribute tokens\\nInput: [1024,10000,8192]\\nOutput: [varies per expert]', 
                         shape='parallelogram', fillcolor='lightyellow')
                
                for gpu in range(8, 16):
                    for expert in range(8):
                        expert_id = (gpu-8) * 2 + (expert % 2) * 8 + expert // 2
                        dot.node(f'layer2_expert{expert_id}_gpu{gpu}', 
                                 f'Expert {expert_id}\\nGPU: {gpu}\\nInput: [batch, seq, 8192]\\nOutput: [batch, seq, 8192]', 
                                 shape='rectangle', fillcolor='lightgreen')
                
                dot.node('layer2_expert_agg', 'Expert Aggregation\\nGPU: 8-15\\nInput: [varies per expert]\\nOutput: [1024,10000,8192]', 
                         shape='parallelogram', fillcolor='lightyellow')
                
                dot.node('layer2_moe_residual', 'Residual Add\\nInput: [1024,10000,8192], [1024,10000,8192]\\nOutput: [1024,10000,8192]', 
                         shape='parallelogram', fillcolor='lightyellow')
        
        # Layer 3
        with c1.subgraph(name='cluster_layer_3') as layer3:
            layer3.attr(label='Layer 3', style='dotted')
            
            # Attention block for layer 3
            with layer3.subgraph(name='cluster_layer3_attention') as attn3:
                attn3.attr(label='Multi-Head Attention', style='rounded')
                
                for i in range(8, 16):
                    dot.node(f'layer3_qkv_gpu{i}', f'QKV Projection\\nGPU: {i}\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,16,512]', 
                             shape='rectangle', fillcolor='lightgreen')
                    
                for i in range(8, 16):
                    dot.node(f'layer3_attn_gpu{i}', f'Attention\\nGPU