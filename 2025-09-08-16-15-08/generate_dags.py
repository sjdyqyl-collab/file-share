import graphviz
import os

def create_baseline_dag():
    """Create baseline DAG with TP=8, PP=2, 4 experts/GPU on 16 GPUs"""
    dot = graphviz.Digraph('baseline_moe_dag', comment='Baseline MoE Deployment')
    dot.attr(rankdir='TB', size='20,30')
    
    # Define global attributes
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Input layer
    dot.node('input', 'Model Input\n[1024, 10000, 8192]', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Pipeline stages (2 stages)
    for stage in range(2):
        with dot.subgraph(name=f'cluster_stage_{stage}') as stage_subgraph:
            stage_subgraph.attr(label=f'Pipeline Stage {stage}', style='rounded', bgcolor='lightgray')
            
            # Each stage has 2 layers
            for layer in range(2):
                layer_id = stage * 2 + layer
                
                with stage_subgraph.subgraph(name=f'cluster_layer_{layer_id}') as layer_subgraph:
                    layer_subgraph.attr(label=f'Layer {layer_id}', style='dashed')
                    
                    # MHA for this layer
                    mha_name = f'mha_{layer_id}'
                    layer_subgraph.node(mha_name, f'MHA Layer {layer_id}\n[1024, 10000, 8192]\nTP=8', 
                                       shape='rectangle', style='filled', fillcolor='lightgreen')
                    
                    # Residual connection
                    res_name = f'res_mha_{layer_id}'
                    layer_subgraph.node(res_name, f'Residual Add\n[1024, 10000, 8192]', 
                                       shape='parallelogram', style='filled', fillcolor='yellow')
                    
                    # MoE layer - 4 experts per GPU, 8 GPUs for TP
                    # 16 experts total, distributed across 4 GPUs per stage
                    for expert_group in range(4):  # 4 groups of 4 experts
                        gpu_id = stage * 8 + expert_group * 2  # 8 GPUs per stage
                        
                        # Expert group on 2 GPUs (each GPU has 4 experts)
                        for gpu_offset in range(2):
                            actual_gpu = gpu_id + gpu_offset
                            expert_start = expert_group * 4 + gpu_offset * 2
                            
                            # Gate for expert selection
                            gate_name = f'gate_{layer_id}_{actual_gpu}'
                            layer_subgraph.node(gate_name, 
                                               f'Gate Layer {layer_id}\nGPU {actual_gpu}\n[1024, 10000, 16]', 
                                               shape='diamond', style='filled', fillcolor='orange')
                            
                            # Expert 1
                            expert1_name = f'expert_{layer_id}_{expert_start}_gpu_{actual_gpu}'
                            layer_subgraph.node(expert1_name, 
                                               f'Expert {expert_start}\nGPU {actual_gpu}\n[1024, 10000, 8192] → [1024, 10000, 32768] → [1024, 10000, 8192]', 
                                               shape='rectangle', style='filled', fillcolor='lightcoral')
                            
                            # Expert 2
                            expert2_name = f'expert_{layer_id}_{expert_start+1}_gpu_{actual_gpu}'
                            layer_subgraph.node(expert2_name, 
                                               f'Expert {expert_start+1}\nGPU {actual_gpu}\n[1024, 10000, 8192] → [1024, 10000, 32768] → [1024, 10000, 8192]', 
                                               shape='rectangle', style='filled', fillcolor='lightcoral')
                            
                            # Expert aggregation
                            agg_name = f'agg_{layer_id}_{actual_gpu}'
                            layer_subgraph.node(agg_name, 
                                               f'Expert Aggregation\nGPU {actual_gpu}\n[1024, 10000, 8192]', 
                                               shape='parallelogram', style='filled', fillcolor='lightyellow')
                    
                    # Final residual
                    final_res_name = f'final_res_{layer_id}'
                    layer_subgraph.node(final_res_name, f'Final Residual\nLayer {layer_id}\n[1024, 10000, 8192]', 
                                       shape='parallelogram', style='filled', fillcolor='yellow')
    
    # Output layer
    dot.node('output', 'Model Output\n[1024, 10000, 8192]', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connect all nodes
    dot.edge('input', 'mha_0')
    
    for layer_id in range(4):
        # MHA connections
        if layer_id > 0:
            prev_layer = layer_id - 1
            dot.edge(f'final_res_{prev_layer}', f'mha_{layer_id}')
        
        # Within layer connections
        dot.edge(f'mha_{layer_id}', f'res_mha_{layer_id}')
        
        # Gate connections (simplified)
        for gpu_id in range(16):
            if gpu_id < 8:  # Stage 0
                stage = 0
            else:  # Stage 1
                stage = 1
                gpu_id -= 8
            
            gate_name = f'gate_{layer_id}_{gpu_id}'
            dot.edge(f'res_mha_{layer_id}', gate_name)
            
            # Expert connections
            for expert_offset in range(4):  # 4 experts per GPU
                expert_name = f'expert_{layer_id}_{gpu_id*4+expert_offset}_gpu_{gpu_id}'
                dot.edge(gate_name, expert_name, style='dashed', label=f'route to expert')
                
                # Aggregation
                agg_name = f'agg_{layer_id}_{gpu_id}'
                dot.edge(expert_name, agg_name)
        
        # Final connections
        dot.edge(f'agg_{layer_id}_0', f'final_res_{layer_id}')  # Simplified
        dot.edge(f'res_mha_{layer_id}', f'final_res_{layer_id}')  # Residual
    
    dot.edge('final_res_3', 'output')
    
    return dot

def create_proposed_dag():
    """Create proposed DAG with 1 expert/GPU, EP=16 on 64 GPUs"""
    dot = graphviz.Digraph('proposed_moe_dag', comment='Proposed MoE Deployment (1 Expert/GPU)')
    dot.attr(rankdir='TB', size='30,40')
    
    # Define global attributes
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Input layer
    dot.node('input', 'Model Input\n[1024, 10000, 8192]', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # 4 layers, each with 16 experts on separate GPUs
    for layer in range(4):
        with dot.subgraph(name=f'cluster_layer_{layer}') as layer_subgraph:
            layer_subgraph.attr(label=f'Layer {layer} - 16 Experts on 16 GPUs (EP=16)', style='rounded', bgcolor='lightgray')
            
            # MHA for this layer (can use tensor parallelism across multiple GPUs)
            mha_name = f'mha_{layer}'
            layer_subgraph.node(mha_name, f'MHA Layer {layer}\n[1024, 10000, 8192]\nTP=8', 
                               shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # Residual connection
            res_name = f'res_mha_{layer}'
            layer_subgraph.node(res_name, f'Residual Add\n[1024, 10000, 8192]', 
                               shape='parallelogram', style='filled', fillcolor='yellow')
            
            # Gate for expert selection
            gate_name = f'gate_{layer}'
            layer_subgraph.node(gate_name, f'Gate Layer {layer}\n[1024, 10000, 16]', 
                               shape='diamond', style='filled', fillcolor='orange')
            
            # Create 16 experts, each on a separate GPU
            for expert_id in range(16):
                gpu_id = layer * 16 + expert_id  # 16 GPUs per layer
                
                # Communication node for token routing
                comm_name = f'comm_{layer}_{expert_id}'
                layer_subgraph.node(comm_name, 
                                   f'Token Routing\nExpert {expert_id}\nGPU {gpu_id}\nAsync Transfer', 
                                   shape='ellipse', style='filled', fillcolor='lightcyan')
                
                # Expert on dedicated GPU
                expert_name = f'expert_{layer}_{expert_id}_gpu_{gpu_id}'
                layer_subgraph.node(expert_name, 
                                   f'Expert {expert_id}\nGPU {gpu_id}\n[1024, 10000, 8192] → [1024, 10000, 32768] → [1024, 10000, 8192]', 
                                   shape='rectangle', style='filled', fillcolor='lightcoral')
                
                # Communication back
                back_comm_name = f'back_comm_{layer}_{expert_id}'
                layer_subgraph.node(back_comm_name, 
                                   f'Token Return\nExpert {expert_id}\nGPU {gpu_id}\nAsync Transfer', 
                                   shape='ellipse', style='filled', fillcolor='lightcyan')
            
            # Expert aggregation
            agg_name = f'agg_{layer}'
            layer_subgraph.node(agg_name, f'Expert Aggregation\n[1024, 10000, 8192]', 
                               shape='parallelogram', style='filled', fillcolor='lightyellow')
            
            # Final residual
            final_res_name = f'final_res_{layer}'
            layer_subgraph.node(final_res_name, f'Final Residual\nLayer {layer}\n[1024, 10000, 8192]', 
                               shape='parallelogram', style='filled', fillcolor='yellow')
    
    # Output layer
    dot.node('output', 'Model Output\n[1024, 10000, 8192]', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connect all nodes
    dot.edge('input', 'mha_0')
    
    for layer in range(4):
        # MHA connections
        if layer > 0:
            prev_layer = layer - 1
            dot.edge(f'final_res_{prev_layer}', f'mha_{layer}')
        
        # Within layer connections
        dot.edge(f'mha_{layer}', f'res_mha_{layer}')
        dot.edge(f'res_mha_{layer}', f'gate_{layer}')
        
        # Expert routing connections
        for expert_id in range(16):
            gpu_id = layer * 16 + expert_id
            
            comm_name = f'comm_{layer}_{expert_id}'
            expert_name = f'expert_{layer}_{expert_id}_gpu_{gpu_id}'
            back_comm_name = f'back_comm_{layer}_{expert_id}'
            
            # Gate to communication
            dot.edge(f'gate_{layer}', comm_name, style='dashed', label=f'route tokens')
            
            # Communication to expert
            dot.edge(comm_name, expert_name)
            
            # Expert to return communication
            dot.edge(expert_name, back_comm_name)
            
            # Return to aggregation
            dot.edge(back_comm_name, f'agg_{layer}')
        
        # Final connections
        dot.edge(f'agg_{layer}', f'final_res_{layer}')
        dot.edge(f'res_mha_{layer}', f'final_res_{layer}')  # Residual
    
    dot.edge('final_res_3', 'output')
    
    return dot

if __name__ == "__main__":
    # Create output directory
    os.makedirs('/home/wzc/data/file-share/2025-09-08-16-15-08', exist_ok=True)
    
    # Generate baseline DAG
    print("Generating baseline DAG...")
    baseline_dag = create_baseline_dag()
    baseline_dag.render('/home/wzc/data/file-share/2025-09-08-16-15-08/baseline_moe_dag', format='svg', cleanup=False)
    
    # Generate proposed DAG
    print("Generating proposed DAG...")
    proposed_dag = create_proposed_dag()
    proposed_dag.render('/home/wzc/data/file-share/2025-09-08-16-15-08/proposed_moe_dag', format='svg', cleanup=False)
    
    # Save DOT files
    with open('/home/wzc/data/file-share/2025-09-08-16-15-08/baseline_moe_dag.dot', 'w') as f:
        f.write(baseline_dag.source)
    
    with open('/home/wzc/data/file-share/2025-09-08-16-15-08/proposed_moe_dag.dot', 'w') as f:
        f.write(proposed_dag.source)
    
    print("DAGs generated successfully!")
    print("Files saved:")
    print("- /home/wzc/data/file-share/2025-09-08-16-15-08/baseline_moe_dag.svg")
    print("- /home/wzc/data/file-share/2025-09-08-16-15-08/proposed_moe_dag.svg")
    print("- /home/wzc/data/file-share/2025-09-08-16-15-08/baseline_moe_dag.dot")
    print("- /home/wzc/data/file-share/2025-09-08-16-15-08/proposed_moe_dag.dot")