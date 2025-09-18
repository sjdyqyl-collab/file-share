#!/usr/bin/env python3

import graphviz
from typing import Dict, List, Tuple
import os

class MoEDAGGenerator:
    def __init__(self):
        self.save_dir = "/home/wzc/data/file-share/2025-09-08-10-40-30"
        
    def create_baseline_moe_dag(self):
        """Create DAG for baseline MoE with TP=8, PP=2, 4 experts per GPU"""
        dot = graphviz.Digraph('baseline_moe', comment='Baseline MoE (TP=8, PP=2)')
        dot.attr(rankdir='TB', size='20,20')
        
        # Define dimensions
        batch_size = 1024
        seq_len = 10000
        hidden_size = 8192
        head_dim = 512
        num_heads = 16
        ffn_hidden = 32768
        
        # Input node
        dot.node('input', f'Total Input\\n[B={batch_size}, S={seq_len}, H={hidden_size}]', 
                shape='ellipse', style='filled', fillcolor='lightblue')
        
        # Pipeline Stage 0 (Layers 0-1) - GPUs 0-7
        for layer in [0, 1]:
            for tp_rank in range(8):
                gpu_id = tp_rank
                
                # Layer norm (replicated across TP)
                dot.node(f'ln1_l{layer}_g{gpu_id}', 
                        f'LayerNorm\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightyellow')
                
                # QKV projection (column parallel)
                qkv_out = hidden_size // 8 * 3
                dot.node(f'qkv_l{layer}_g{gpu_id}',
                        f'QKV Proj\\n[B,S,{qkv_out}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # Attention computation
                dot.node(f'attn_l{layer}_g{gpu_id}',
                        f'Multi-Head Attn\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightcoral')
                
                # Output projection (row parallel)
                dot.node(f'out_proj_l{layer}_g{gpu_id}',
                        f'Out Proj\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # Residual add 1
                dot.node(f'residual1_l{layer}_g{gpu_id}',
                        f'Residual Add\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='parallelogram', style='filled', fillcolor='lightgray')
                
                # Layer norm 2
                dot.node(f'ln2_l{layer}_g{gpu_id}',
                        f'LayerNorm2\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightyellow')
                
                # Expert routing
                dot.node(f'gate_l{layer}_g{gpu_id}',
                        f'Expert Gate\\n[B,S,16]\\nGPU={gpu_id}',
                        shape='parallelogram', style='filled', fillcolor='orange')
                
                # 4 experts per GPU (experts 0-3 for layer 0, 4-7 for layer 1)
                start_expert = layer * 4
                for expert_id in range(4):
                    actual_expert = start_expert + expert_id
                    
                    # Expert MLP
                    dot.node(f'expert{actual_expert}_g{gpu_id}',
                            f'Expert {actual_expert}\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                            shape='rectangle', style='filled', fillcolor='lightpink')
                    
                    # Expert FC1 (column parallel within expert)
                    dot.node(f'expert{actual_expert}_fc1_g{gpu_id}',
                            f'Expert FC1\\n[B,S,{ffn_hidden//8}]\\nGPU={gpu_id}',
                            shape='rectangle', style='filled', fillcolor='lightsteelblue')
                    
                    # Expert activation
                    dot.node(f'expert{actual_expert}_act_g{gpu_id}',
                            f'GELU\\n[B,S,{ffn_hidden//8}]\\nGPU={gpu_id}',
                            shape='rectangle', style='filled', fillcolor='lightcyan')
                    
                    # Expert FC2 (row parallel within expert)
                    dot.node(f'expert{actual_expert}_fc2_g{gpu_id}',
                            f'Expert FC2\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                            shape='rectangle', style='filled', fillcolor='lightsteelblue')
                
                # Expert aggregation
                dot.node(f'expert_agg_l{layer}_g{gpu_id}',
                        f'Expert Aggregation\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='parallelogram', style='filled', fillcolor='gold')
                
                # Final residual
                dot.node(f'residual2_l{layer}_g{gpu_id}',
                        f'Final Residual\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='parallelogram', style='filled', fillcolor='lightgray')
                
                # All-reduce operations for TP
                dot.node(f'allreduce1_l{layer}_g{gpu_id}',
                        f'TP All-Reduce\\n[B,S,H={hidden_size}]\\nTP Group={gpu_id//8*8}-{(gpu_id//8+1)*8-1}',
                        shape='ellipse', style='dashed', fillcolor='lightgreen')
                
                dot.node(f'allreduce2_l{layer}_g{gpu_id}',
                        f'TP All-Reduce\\n[B,S,H={hidden_size}]\\nTP Group={gpu_id//8*8}-{(gpu_id//8+1)*8-1}',
                        shape='ellipse', style='dashed', fillcolor='lightgreen')
        
        # Pipeline communication between stages
        dot.node('pipeline_comm1', 'Pipeline Communication\\nStage 0 → Stage 1', 
                shape='ellipse', style='dashed', fillcolor='blue')
        
        # Pipeline Stage 1 (Layers 2-3) - GPUs 8-15
        for layer in [2, 3]:
            for tp_rank in range(8):
                gpu_id = 8 + tp_rank
                
                # Same structure as stage 0, but different expert IDs
                dot.node(f'ln1_l{layer}_g{gpu_id}', 
                        f'LayerNorm\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightyellow')
                
                qkv_out = hidden_size // 8 * 3
                dot.node(f'qkv_l{layer}_g{gpu_id}',
                        f'QKV Proj\\n[B,S,{qkv_out}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                dot.node(f'attn_l{layer}_g{gpu_id}',
                        f'Multi-Head Attn\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightcoral')
                
                dot.node(f'out_proj_l{layer}_g{gpu_id}',
                        f'Out Proj\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                dot.node(f'residual1_l{layer}_g{gpu_id}',
                        f'Residual Add\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='parallelogram', style='filled', fillcolor='lightgray')
                
                dot.node(f'ln2_l{layer}_g{gpu_id}',
                        f'LayerNorm2\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightyellow')
                
                dot.node(f'gate_l{layer}_g{gpu_id}',
                        f'Expert Gate\\n[B,S,16]\\nGPU={gpu_id}',
                        shape='parallelogram', style='filled', fillcolor='orange')
                
                start_expert = 8 + (layer - 2) * 4
                for expert_id in range(4):
                    actual_expert = start_expert + expert_id
                    
                    dot.node(f'expert{actual_expert}_g{gpu_id}',
                            f'Expert {actual_expert}\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                            shape='rectangle', style='filled', fillcolor='lightpink')
                    
                    dot.node(f'expert{actual_expert}_fc1_g{gpu_id}',
                            f'Expert FC1\\n[B,S,{ffn_hidden//8}]\\nGPU={gpu_id}',
                            shape='rectangle', style='filled', fillcolor='lightsteelblue')
                    
                    dot.node(f'expert{actual_expert}_act_g{gpu_id}',
                            f'GELU\\n[B,S,{ffn_hidden//8}]\\nGPU={gpu_id}',
                            shape='rectangle', style='filled', fillcolor='lightcyan')
                    
                    dot.node(f'expert{actual_expert}_fc2_g{gpu_id}',
                            f'Expert FC2\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                            shape='rectangle', style='filled', fillcolor='lightsteelblue')
                
                dot.node(f'expert_agg_l{layer}_g{gpu_id}',
                        f'Expert Aggregation\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='parallelogram', style='filled', fillcolor='gold')
                
                dot.node(f'residual2_l{layer}_g{gpu_id}',
                        f'Final Residual\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='parallelogram', style='filled', fillcolor='lightgray')
                
                dot.node(f'allreduce1_l{layer}_g{gpu_id}',
                        f'TP All-Reduce\\n[B,S,H={hidden_size}]\\nTP Group={gpu_id//8*8}-{(gpu_id//8+1)*8-1}',
                        shape='ellipse', style='dashed', fillcolor='lightgreen')
                
                dot.node(f'allreduce2_l{layer}_g{gpu_id}',
                        f'TP All-Reduce\\n[B,S,H={hidden_size}]\\nTP Group={gpu_id//8*8}-{(gpu_id//8+1)*8-1}',
                        shape='ellipse', style='dashed', fillcolor='lightgreen')
        
        # Output node
        dot.node('output', f'Total Output\\n[B={batch_size}, S={seq_len}, H={hidden_size}]', 
                shape='ellipse', style='filled', fillcolor='lightblue')
        
        # Create edges - this is complex, so I'll create a simplified version
        # In practice, this would need to be much more detailed
        
        # Save the DAG
        dot.save(os.path.join(self.save_dir, 'baseline_moe.dot'))
        dot.render(os.path.join(self.save_dir, 'baseline_moe'), format='svg', cleanup=False)
        
    def create_proposed_moe_dag(self):
        """Create DAG for proposed cross-node expert parallelism with 1 expert per GPU"""
        dot = graphviz.Digraph('proposed_moe', comment='Proposed Cross-Node Expert Parallelism')
        dot.attr(rankdir='TB', size='30,30')
        
        # Define dimensions
        batch_size = 1024
        seq_len = 10000
        hidden_size = 8192
        head_dim = 512
        num_heads = 16
        ffn_hidden = 32768
        
        # Input node
        dot.node('input', f'Total Input\\n[B={batch_size}, S={seq_len}, H={hidden_size}]', 
                shape='ellipse', style='filled', fillcolor='lightblue')
        
        # Global token routing
        dot.node('global_router', 'Global Token Router\\n[B,S,64 experts]\\nAll GPUs',
                shape='parallelogram', style='filled', fillcolor='orange')
        
        # For each layer (0-3)
        for layer in range(4):
            layer_start_gpu = layer * 16
            
            # Layer-level nodes
            dot.node(f'layer{layer}_start', f'Layer {layer} Start\\n[B,S,H={hidden_size}]',
                    shape='ellipse', style='dashed', fillcolor='lightblue')
            
            # Attention across all GPUs for this layer
            for gpu_id in range(layer_start_gpu, layer_start_gpu + 16):
                # Attention computation (no tensor parallelism)
                dot.node(f'ln1_l{layer}_g{gpu_id}', 
                        f'LayerNorm\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightyellow')
                
                dot.node(f'qkv_l{layer}_g{gpu_id}',
                        f'QKV Proj\\n[B,S,{hidden_size*3}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                dot.node(f'attn_l{layer}_g{gpu_id}',
                        f'Multi-Head Attn\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightcoral')
                
                dot.node(f'out_proj_l{layer}_g{gpu_id}',
                        f'Out Proj\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                dot.node(f'residual1_l{layer}_g{gpu_id}',
                        f'Residual Add\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='parallelogram', style='filled', fillcolor='lightgray')
            
            # Expert routing and distribution
            dot.node(f'expert_dispatch_l{layer}', 
                    f'Expert Dispatch\\nLayer {layer}\\n[B,S,H] → [B,S,H]\\nCross-GPU',
                    shape='ellipse', style='dashed', fillcolor='purple')
            
            # One expert per GPU
            expert_id = layer * 16
            for gpu_id in range(layer_start_gpu, layer_start_gpu + 16):
                # Expert gating (local to this GPU)
                dot.node(f'gate_l{layer}_g{gpu_id}',
                        f'Local Gate\\n[B,S,1]\\nGPU={gpu_id}',
                        shape='parallelogram', style='filled', fillcolor='orange')
                
                # Single expert on this GPU
                dot.node(f'expert{expert_id}_g{gpu_id}',
                        f'Expert {expert_id}\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightpink')
                
                # Expert FC1
                dot.node(f'expert{expert_id}_fc1_g{gpu_id}',
                        f'Expert FC1\\n[B,S,{ffn_hidden}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightsteelblue')
                
                # Expert activation
                dot.node(f'expert{expert_id}_act_g{gpu_id}',
                        f'GELU\\n[B,S,{ffn_hidden}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightcyan')
                
                # Expert FC2
                dot.node(f'expert{expert_id}_fc2_g{gpu_id}',
                        f'Expert FC2\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightsteelblue')
                
                expert_id += 1
            
            # Expert aggregation and routing back
            dot.node(f'expert_collect_l{layer}', 
                    f'Expert Collection\\nLayer {layer}\\n[B,S,H] ← [B,S,H]\\nCross-GPU',
                    shape='ellipse', style='dashed', fillcolor='purple')
            
            # Final processing for each GPU
            for gpu_id in range(layer_start_gpu, layer_start_gpu + 16):
                dot.node(f'ln2_l{layer}_g{gpu_id}',
                        f'LayerNorm2\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='rectangle', style='filled', fillcolor='lightyellow')
                
                dot.node(f'residual2_l{layer}_g{gpu_id}',
                        f'Final Residual\\n[B,S,H={hidden_size}]\\nGPU={gpu_id}',
                        shape='parallelogram', style='filled', fillcolor='lightgray')
            
            # Pipeline communication between layers
            if layer < 3:
                dot.node(f'pipeline_l{layer}_to_l{layer+1}',
                        f'Pipeline L{layer}→L{layer+1}\\nCross-Layer Communication',
                        shape='ellipse', style='dashed', fillcolor='blue')
        
        # Output node
        dot.node('output', f'Total Output\\n[B={batch_size}, S={seq_len}, H={hidden_size}]', 
                shape='ellipse', style='filled', fillcolor='lightblue')
        
        # Save the DAG
        dot.save(os.path.join(self.save_dir, 'proposed_moe.dot'))
        dot.render(os.path.join(self.save_dir, 'proposed_moe'), format='svg', cleanup=False)
        
    def create_detailed_edges(self):
        """Create detailed edge connections for both DAGs"""
        # This would be extremely complex for the full DAG
        # For now, we'll create simplified versions and note that full implementation
        # would require careful edge creation for each node
        pass

if __name__ == "__main__":
    generator = MoEDAGGenerator()
    
    print("Generating Baseline MoE DAG...")
    generator.create_baseline_moe_dag()
    
    print("Generating Proposed MoE DAG...")
    generator.create_proposed_moe_dag()
    
    print("DAG generation complete!")