import graphviz
from typing import Dict, List, Tuple
import os

class MoEDAGGenerator:
    def __init__(self):
        self.colors = {
            'compute': '#E8F4FD',
            'communication': '#FFF2CC',
            'routing': '#E1F5E1',
            'input': '#F0F0F0',
            'output': '#FFE6E6'
        }
    
    def create_baseline_dag(self):
        """Create baseline DAG with TP=8, PP=2, 16 GPUs total"""
        dot = graphviz.Digraph('baseline_moe', comment='Baseline MoE Deployment')
        dot.attr(rankdir='TB', size='20,15')
        
        # Global settings
        dot.attr('node', fontname='Arial', fontsize='10')
        
        # Input node
        dot.node('input', 'Input\\n[1024, seq_len, 8192]', 
                shape='ellipse', style='filled', fillcolor=self.colors['input'])
        
        # Process 4 layers with pipeline parallelism
        for layer in range(4):
            layer_prefix = f'layer{layer}'
            
            # Pipeline stage 0 (layers 0,2 on GPUs 0-7)
            if layer % 2 == 0:
                self._add_layer_baseline(dot, layer, layer_prefix, 0, 8)
            # Pipeline stage 1 (layers 1,3 on GPUs 8-15)  
            else:
                self._add_layer_baseline(dot, layer, layer_prefix, 8, 16)
                
            # Pipeline communication between stages
            if layer > 0:
                prev_layer = layer - 1
                prev_stage = 0 if prev_layer % 2 == 0 else 8
                curr_stage = 8 if layer % 2 == 1 else 0
                
                dot.node(f'pipe_comm_{layer}', f'Pipeline Comm\\nLayer {prev_layer} -> {layer}',
                        shape='ellipse', style='filled, dashed', fillcolor=self.colors['communication'])
                
                # Connect pipeline stages
                for gpu in range(8):
                    dot.edge(f'layer{prev_layer}_output_gpu{prev_stage+gpu}', 
                           f'pipe_comm_{layer}')
                    dot.edge(f'pipe_comm_{layer}', 
                           f'layer{layer}_mha_input_gpu{curr_stage+gpu}')
        
        # Output node
        dot.node('output', 'Output\\n[1024, seq_len, 8192]', 
                shape='ellipse', style='filled', fillcolor=self.colors['output'])
        
        # Connect final layer to output
        final_layer = 3
        final_stage = 8 if final_layer % 2 == 1 else 0
        for gpu in range(8):
            dot.edge(f'layer{final_layer}_output_gpu{final_stage+gpu}', 'output')
            
        return dot
    
    def _add_layer_baseline(self, dot, layer, prefix, gpu_start, gpu_end):
        """Add a complete layer with MHA and MoE for baseline"""
        # MHA section - tensor parallel across 8 GPUs
        for gpu in range(gpu_start, gpu_end):
            gpu_id = gpu - gpu_start
            
            # Input to MHA
            dot.node(f'{prefix}_mha_input_gpu{gpu}', 
                   f'MHA Input\\n[1024, seq_len, 8192]\\nGPU {gpu}',
                   shape='rectangle', style='filled', fillcolor=self.colors['compute'])
            
            # QKV projection (tensor parallel)
            dot.node(f'{prefix}_qkv_gpu{gpu}', 
                   f'QKV Projection\\n[1024, seq_len, 1536]\\nGPU {gpu}',
                   shape='rectangle', style='filled', fillcolor=self.colors['compute'])
            
            # Multi-head attention
            dot.node(f'{prefix}_mha_gpu{gpu}', 
                   f'MHA (2 heads)\\n[1024, seq_len, 1024]\\nGPU {gpu}',
                   shape='rectangle', style='filled', fillcolor=self.colors['compute'])
            
            # Output projection
            dot.node(f'{prefix}_mha_out_gpu{gpu}', 
                   f'MHA Output\\n[1024, seq_len, 8192]\\nGPU {gpu}',
                   shape='rectangle', style='filled', fillcolor=self.colors['compute'])
            
            # Residual connection
            dot.node(f'{prefix}_mha_res_gpu{gpu}', 
                   f'Residual Add\\n[1024, seq_len, 8192]\\nGPU {gpu}',
                   shape='parallelogram', style='filled', fillcolor=self.colors['routing'])
            
            # Layer norm
            dot.node(f'{prefix}_ln1_gpu{gpu}', 
                   f'Layer Norm\\n[1024, seq_len, 8192]\\nGPU {gpu}',
                   shape='rectangle', style='filled', fillcolor=self.colors['compute'])
            
            # MoE section - 4 experts per GPU
            for expert in range(4):
                expert_id = gpu_id * 4 + expert
                dot.node(f'{prefix}_expert{expert_id}_gpu{gpu}', 
                       f'Expert {expert_id}\\nMLP [1024, seq_len, 8192]\\nGPU {gpu}',
                       shape='rectangle', style='filled', fillcolor=self.colors['compute'])
            
            # Gating network
            dot.node(f'{prefix}_gate_gpu{gpu}', 
                   f'Gating Network\\n[1024, seq_len, 16]\\nGPU {gpu}',
                   shape='parallelogram', style='filled', fillcolor=self.colors['routing'])
            
            # Expert aggregation
            dot.node(f'{prefix}_expert_agg_gpu{gpu}', 
                   f'Expert Aggregation\\n[1024, seq_len, 8192]\\nGPU {gpu}',
                   shape='parallelogram', style='filled', fillcolor=self.colors['routing'])
            
            # Output projection
            dot.node(f'{prefix}_mlp_out_gpu{gpu}', 
                   f'MLP Output\\n[1024, seq_len, 8192]\\nGPU {gpu}',
                   shape='rectangle', style='filled', fillcolor=self.colors['compute'])
            
            # Final residual
            dot.node(f'{prefix}_output_gpu{gpu}', 
                   f'Layer Output\\n[1024, seq_len, 8192]\\nGPU {gpu}',
                   shape='rectangle', style='filled', fillcolor=self.colors['compute'])
            
            # Connections within layer
            if layer == 0:
                # Connect input to first layer
                dot.edge('input', f'{prefix}_mha_input_gpu{gpu}')
            
            dot.edge(f'{prefix}_mha_input_gpu{gpu}', f'{prefix}_qkv_gpu{gpu}')
            dot.edge(f'{prefix}_qkv_gpu{gpu}', f'{prefix}_mha_gpu{gpu}')
            dot.edge(f'{prefix}_mha_gpu{gpu}', f'{prefix}_mha_out_gpu{gpu}')
            dot.edge(f'{prefix}_mha_out_gpu{gpu}', f'{prefix}_mha_res_gpu{gpu}')
            dot.edge(f'{prefix}_mha_input_gpu{gpu}', f'{prefix}_mha_res_gpu{gpu}')  # Residual
            dot.edge(f'{prefix}_mha_res_gpu{gpu}', f'{prefix}_ln1_gpu{gpu}')
            
            # Gating to experts (dashed lines)
            dot.edge(f'{prefix}_ln1_gpu{gpu}', f'{prefix}_gate_gpu{gpu}')
            for expert in range(4):
                expert_id = gpu_id * 4 + expert
                dot.edge(f'{prefix}_gate_gpu{gpu}', f'{prefix}_expert{expert_id}_gpu{gpu}', 
                        style='dashed')
                dot.edge(f'{prefix}_ln1_gpu{gpu}', f'{prefix}_expert{expert_id}_gpu{gpu}')
                dot.edge(f'{prefix}_expert{expert_id}_gpu{gpu}', f'{prefix}_expert_agg_gpu{gpu}')
            
            dot.edge(f'{prefix}_gate_gpu{gpu}', f'{prefix}_expert_agg_gpu{gpu}')
            dot.edge(f'{prefix}_expert_agg_gpu{gpu}', f'{prefix}_mlp_out_gpu{gpu}')
            dot.edge(f'{prefix}_mlp_out_gpu{gpu}', f'{prefix}_output_gpu{gpu}')
            dot.edge(f'{prefix}_ln1_gpu{gpu}', f'{prefix}_output_gpu{gpu}')  # Residual
    
    def create_proposed_dag(self):
        """Create proposed DAG with 1 expert per GPU, 64 GPUs total"""
        dot = graphviz.Digraph('proposed_moe', comment='Proposed Large-Scale MoE Deployment')
        dot.attr(rankdir='TB', size='30,20')
        
        # Global settings
        dot.attr('node', fontname='Arial', fontsize='10')
        
        # Input node
        dot.node('input', 'Input\\n[1024, seq_len, 8192]', 
                shape='ellipse', style='filled', fillcolor=self.colors['input'])
        
        # Process 4 layers
        for layer in range(4):
            layer_prefix = f'layer{layer}'
            self._add_layer_proposed(dot, layer, layer_prefix)
            
            # Connect layers
            if layer == 0:
                for gpu in range(64):
                    dot.edge('input', f'{layer_prefix}_mha_input_gpu{gpu}')
            else:
                prev_layer = layer - 1
                prefix_prev = f'layer{prev_layer}'
                for gpu in range(64):
                    dot.edge(f'{prefix_prev}_output_gpu{gpu}', 
                           f'{layer_prefix}_mha_input_gpu{gpu}')
        
        # Output node
        dot.node('output', 'Output\\n[1024, seq_len, 8192]', 
                shape='ellipse', style='filled', fillcolor=self.colors['output'])
        
        # Connect final layer to output
        final_layer = 3
        prefix_final = f'layer{final_layer}'
        for gpu in range(64):
            dot.edge(f'{prefix_final}_output_gpu{gpu}', 'output')
            
        return dot
    
    def _add_layer_proposed(self, dot, layer, prefix):
        """Add a complete layer with MHA and single expert per GPU"""
        
        # MHA section - replicated across all GPUs
        for gpu in range(64):
            # Input to MHA
            dot.node(f'{prefix}_mha_input_gpu{gpu}', 
                   f'MHA Input\\n[1024, seq_len, 8192]\\nGPU {gpu}',
                   shape='rectangle', style='filled', fillcolor=self.colors['compute'])
            
            # QKV projection
            dot.node(f'{prefix}_qkv_gpu{gpu}', 
                   f'QKV Projection\\n[1024, seq_len, 1536]\\nGPU {gpu}',
                   shape='rectangle', style='filled', fillcolor=self.colors['compute'])
            
            # Multi-head attention (full 16 heads on each GPU)
            dot.node(f'{prefix}_mha_gpu{gpu}', 
                   f'MHA (16 heads)\\n[1024, seq_len, 8192]\\nGPU {gpu}',
                   shape='rectangle', style='filled', fillcolor=self.colors['compute'])
            
            # Output projection
            dot.node(f'{prefix}_mha_out_gpu{gpu}', 
                   f'MHA Output\\n[1024, seq_len, 8192]\\nGPU {gpu}',
                   shape='rectangle', style='filled', fillcolor=self.colors['compute'])
            
            # Residual connection
            dot.node(f'{prefix}_mha_res_gpu{gpu}', 
                   f'Residual Add\\n[1024, seq_len, 8192]\\nGPU {gpu}',
                   shape='parallelogram', style='filled', fillcolor=self.colors['routing'])
            
            # Layer norm
            dot.node(f'{prefix}_ln1_gpu{gpu}', 
                   f'Layer Norm\\n[1024, seq_len, 8192]\\nGPU {gpu}',
                   shape='rectangle', style='filled', fillcolor=self.colors['compute'])
            
            # Expert - one per GPU (expert ID = GPU ID for this layer)
            expert_id = layer * 16 + (gpu % 16)  # 16 experts per layer
            dot.node(f'{prefix}_expert{expert_id}_gpu{gpu}', 
                   f'Expert {expert_id}\\nMLP [1024, seq_len, 8192]\\nGPU {gpu}',
                   shape='rectangle', style='filled', fillcolor=self.colors['compute'])
            
            # Gating network (global view)
            dot.node(f'{prefix}_gate_gpu{gpu}', 
                   f'Gating Network\\n[1024, seq_len, 16]\\nGPU {gpu}',
                   shape='parallelogram', style='filled', fillcolor=self.colors['routing'])
            
            # Expert aggregation
            dot.node(f'{prefix}_expert_agg_gpu{gpu}', 
                   f'Expert Aggregation\\n[1024, seq_len, 8192]\\nGPU {gpu}',
                   shape='parallelogram', style='filled', fillcolor=self.colors['routing'])
            
            # Output projection
            dot.node(f'{prefix}_mlp_out_gpu{gpu}', 
                   f'MLP Output\\n[1024, seq_len, 8192]\\nGPU {gpu}',
                   shape='rectangle', style='filled', fillcolor=self.colors['compute'])
            
            # Final residual
            dot.node(f'{prefix}_output_gpu{gpu}', 
                   f'Layer Output\\n[1024, seq_len, 8192]\\nGPU {gpu}',
                   shape='rectangle', style='filled', fillcolor=self.colors['compute'])
            
            # Connections within layer
            dot.edge(f'{prefix}_mha_input_gpu{gpu}', f'{prefix}_qkv_gpu{gpu}')
            dot.edge(f'{prefix}_qkv_gpu{gpu}', f'{prefix}_mha_gpu{gpu}')
            dot.edge(f'{prefix}_mha_gpu{gpu}', f'{prefix}_mha_out_gpu{gpu}')
            dot.edge(f'{prefix}_mha_out_gpu{gpu}', f'{prefix}_mha_res_gpu{gpu}')
            dot.edge(f'{prefix}_mha_input_gpu{gpu}', f'{prefix}_mha_res_gpu{gpu}')  # Residual
            dot.edge(f'{prefix}_mha_res_gpu{gpu}', f'{prefix}_ln1_gpu{gpu}')
            
            # Gating to expert (dashed line)
            dot.edge(f'{prefix}_ln1_gpu{gpu}', f'{prefix}_gate_gpu{gpu}')
            dot.edge(f'{prefix}_gate_gpu{gpu}', f'{prefix}_expert{expert_id}_gpu{gpu}', 
                    style='dashed')
            dot.edge(f'{prefix}_ln1_gpu{gpu}', f'{prefix}_expert{expert_id}_gpu{gpu}')
            dot.edge(f'{prefix}_expert{expert_id}_gpu{gpu}', f'{prefix}_expert_agg_gpu{gpu}')
            
            dot.edge(f'{prefix}_gate_gpu{gpu}', f'{prefix}_expert_agg_gpu{gpu}')
            dot.edge(f'{prefix}_expert_agg_gpu{gpu}', f'{prefix}_mlp_out_gpu{gpu}')
            dot.edge(f'{prefix}_mlp_out_gpu{gpu}', f'{prefix}_output_gpu{gpu}')
            dot.edge(f'{prefix}_ln1_gpu{gpu}', f'{prefix}_output_gpu{gpu}')  # Residual

def main():
    generator = MoEDAGGenerator()
    
    # Generate baseline DAG
    baseline_dag = generator.create_baseline_dag()
    baseline_dag.render('/home/wzc/data/file-share/2025-09-04-15-56-46/baseline_moe', 
                       format='svg', cleanup=False)
    
    # Generate proposed DAG
    proposed_dag = generator.create_proposed_dag()
    proposed_dag.render('/home/wzc/data/file-share/2025-09-04-15-56-46/proposed_moe', 
                       format='svg', cleanup=False)
    
    # Save DOT files
    with open('/home/wzc/data/file-share/2025-09-04-15-56-46/baseline_moe.dot', 'w') as f:
        f.write(baseline_dag.source)
    
    with open('/home/wzc/data/file-share/2025-09-04-15-56-46/proposed_moe.dot', 'w') as f:
        f.write(proposed_dag.source)
    
    print("DAGs generated successfully!")
    print("Files created:")
    print("- baseline_moe.svg")
    print("- baseline_moe.dot") 
    print("- proposed_moe.svg")
    print("- proposed_moe.dot")

if __name__ == "__main__":
    main()