#!/usr/bin/env python3

import graphviz
import json
import os

def create_baseline_dag_detailed():
    """Create detailed baseline DAG for 16 GPUs with TP=8, PP=2, 4 experts/GPU"""
    dot = graphviz.Digraph('baseline_moe_16gpus', format='svg')
    dot.attr(rankdir='TB', size='50,40', ranksep='1.5', nodesep='0.8')
    
    # Model dimensions
    batch_size = 1024
    seq_len = 10000
    hidden_dim = 8192
    ffn_hidden = 32768
    num_experts = 16
    
    # Input
    dot.node('input', f'Total Input\\n[Batch: {batch_size}×{seq_len},\\nHidden: {hidden_dim}]', 
             shape='parallelogram', fillcolor='lightgreen', style='filled')
    
    # Process all 4 layers
    for layer_id in range(4):
        stage = layer_id % 2
        gpu_base = 0 if stage == 0 else 8
        
        # Create layer cluster
        layer_name = f'layer_{layer_id}'
        with dot.subgraph(name=f'cluster_{layer_name}') as layer:
            layer.attr(label=f'MoE Layer {layer_id}\\n(Pipeline Stage {stage}, GPUs {gpu_base}-{gpu_base+7})', 
                      style='dashed', color='blue', fillcolor='aliceblue')
            
            # Gating network
            gate_name = f'gate_{layer_id}'
            layer.node(gate_name, 
                      f'Gating Network\\nLayer {layer_id}\\n[Input: {hidden_dim},\\nOutput: {num_experts} experts]\\nGPU {gpu_base}',
                      shape='diamond', fillcolor='yellow', style='filled')
            
            # Expert clusters
            for expert_id in range(num_experts):
                gpu_id = gpu_base + (expert_id // 4)
                expert_name = f'expert_{layer_id}_{expert_id}'
                
                with layer.subgraph(name=f'cluster_{expert_name}') as expert:
                    expert.attr(label=f'Expert {expert_id}\\nGPU {gpu_id}', 
                               style='rounded', color='red')
                    
                    # Expert MLP structure
                    expert.node(f'{expert_name}_linear1', 
                               f'Linear 1\\n[{hidden_dim}→{ffn_hidden}]\\nGPU {gpu_id}',
                               shape='rectangle', fillcolor='lightcoral', style='filled')
                    expert.node(f'{expert_name}_gelu', 
                               f'GELU\\n[{ffn_hidden}]\\nGPU {gpu_id}',
                               shape='ellipse', fillcolor='lightblue', style='filled')
                    expert.node(f'{expert_name}_linear2', 
                               f'Linear 2\\n[{ffn_hidden}→{hidden_dim}]\\nGPU {gpu_id}',
                               shape='rectangle', fillcolor='lightcoral', style='filled')
                    
                    # Connect expert components
                    expert.edge(f'{expert_name}_linear1', f'{expert_name}_gelu')
                    expert.edge(f'{expert_name}_gelu', f'{expert_name}_linear2')
            
            # Expert aggregation
            agg_name = f'expert_agg_{layer_id}'
            layer.node(agg_name, 
                      f'Expert Aggregation\\nLayer {layer_id}\\n[Combine {num_experts} experts]',
                      shape='parallelogram', fillcolor='purple', style='filled')
            
            # Layer norm
            norm_name = f'layer_norm_{layer_id}'
            layer.node(norm_name, 
                      f'Layer Norm\\nLayer {layer_id}\\n[{hidden_dim}]',
                      shape='ellipse', fillcolor='lightgreen', style='filled')
            
            # Add edges within layer
            # Gating to experts
            for expert_id in range(num_experts):
                expert_name = f'expert_{layer_id}_{expert_id}'
                layer.edge(gate_name, f'{expert_name}_linear1', 
                          style='dashed', label='route')
                layer.edge(f'{expert_name}_linear2', agg_name)
            
            layer.edge(agg_name, norm_name)
    
    # Pipeline communication
    for layer_id in range(3):
        next_layer = layer_id + 1
        dot.edge(f'layer_norm_{layer_id}', f'gate_{next_layer}')
    
    # Output
    dot.node('output', f'Final Output\\n[Batch: {batch_size}×{seq_len},\\nHidden: {hidden_dim}]', 
             shape='parallelogram', fillcolor='lightgreen', style='filled')
    dot.edge('layer_norm_3', 'output')
    
    return dot

def create_proposed_dag_detailed():
    """Create detailed proposed DAG for 64 GPUs with EP=16, 1 expert/GPU"""
    dot = graphviz.Digraph('proposed_moe_64gpus', format='svg')
    dot.attr(rankdir='TB', size='60,50', ranksep='2.0', nodesep='1.0')
    
    # Model dimensions
    batch_size = 1024
    seq_len = 10000
    hidden_dim = 8192
    ffn_hidden = 32768
    num_experts = 16
    
    # Input
    dot.node('input', f'Total Input\\n[Batch: {batch_size}×{seq_len},\\nHidden: {hidden_dim}]', 
             shape='parallelogram', fillcolor='lightgreen', style='filled')
    
    # Process all 4 layers
    for layer_id in range(4):
        pipeline_stage = layer_id
        gpu_base = layer_id * 16
        
        # Create layer cluster
        layer_name = f'layer_{layer_id}'
        with dot.subgraph(name=f'cluster_{layer_name}') as layer:
            layer.attr(label=f'MoE Layer {layer_id}\\n(Pipeline Stage {pipeline_stage}, GPUs {gpu_base}-{gpu_base+15})', 
                      style='dashed', color='blue', fillcolor='aliceblue')
            
            # Gating network
            gate_name = f'gate_{layer_id}'
            layer.node(gate_name, 
                      f'Gating Network\\nLayer {layer_id}\\n[Input: {hidden_dim},\\nOutput: {num_experts} experts]\\nGPU {gpu_base}',
                      shape='diamond', fillcolor='yellow', style='filled')
            
            # Token router
            router_name = f'router_{layer_id}'
            layer.node(router_name, 
                      f'Token Router\\nLayer {layer_id}\\n[Cross-node routing]',
                      shape='parallelogram', fillcolor='orange', style='filled')
            
            # Expert placement (1 expert per GPU)
            for expert_id in range(num_experts):
                gpu_id = gpu_base + expert_id
                expert_name = f'expert_{layer_id}_{expert_id}'
                
                with layer.subgraph(name=f'cluster_{expert_name}') as expert:
                    expert.attr(label=f'Expert {expert_id}\\nGPU {gpu_id}', 
                               style='rounded', color='red')
                    
                    # Expert MLP structure
                    expert.node(f'{expert_name}_linear1', 
                               f'Linear 1\\n[{hidden_dim}→{ffn_hidden}]\\nGPU {gpu_id}',
                               shape='rectangle', fillcolor='lightcoral', style='filled')
                    expert.node(f'{expert_name}_gelu', 
                               f'GELU\\n[{ffn_hidden}]\\nGPU {gpu_id}',
                               shape='ellipse', fillcolor='lightblue', style='filled')
                    expert.node(f'{expert_name}_linear2', 
                               f'Linear 2\\n[{ffn_hidden}→{hidden_dim}]\\nGPU {gpu_id}',
                               shape='rectangle', fillcolor='lightcoral', style='filled')
                    
                    # Connect expert components
                    expert.edge(f'{expert_name}_linear1', f'{expert_name}_gelu')
                    expert.edge(f'{expert_name}_gelu', f'{expert_name}_linear2')
            
            # Token aggregator
            aggregator_name = f'aggregator_{layer_id}'
            layer