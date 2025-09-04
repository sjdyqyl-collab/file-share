import graphviz
from graphviz import Digraph

# Create baseline DAG with Tensor Parallelism + Pipeline Parallelism
def create_baseline_dag():
    dot = Digraph(comment='Baseline: Tensor Parallelism + Pipeline Parallelism')
    dot.attr(rankdir='TB', size='20,30')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input\n(B=1024, L=1024, d_model=8192)\nAll GPUs', shape='ellipse', fillcolor='lightgreen')
    
    # Pipeline Stage 0 (Devices 0-7, Layers 0-1)
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0\nDevices 0-7', style='dashed', color='blue')
        
        # Layer 0
        stage0.node('layer0_norm1', 'LayerNorm\n(B=1024, L=1024, d_model=8192)\nDevices 0-7', shape='parallelogram')
        stage0.node('layer0_q_proj', 'Q Projection\n(B=1024, L=1024, d_model=8192) -> (B=1024, L=1024, d_model=8192)\nTP across 8 devices\nEach: (B=1024, L=1024, d_model=1024)')
        stage0.node('layer0_k_proj', 'K Projection\n(B=1024, L=1024, d_model=8192) -> (B=1024, L=1024, d_model=8192)\nTP across 8 devices\nEach: (B=1024, L=1024, d_model=1024)')
        stage0.node('layer0_v_proj', 'V Projection\n(B=1024, L=1024, d_model=8192) -> (B=1024, L=1024, d_model=8192)\nTP across 8 devices\nEach: (B=1024, L=1024, d_model=1024)')
        
        stage0.node('layer0_mha', 'Multi-Head Attention\nTP across 8 devices\nEach head: 2 heads/device\nLocal: (B=1024, L=1024, d_model=1024)')
        stage0.node('layer0_residual1', 'Residual Add\n(B=1024, L=1024, d_model=8192)\nDevices 0-7', shape='ellipse')
        
        stage0.node('layer0_norm2', 'LayerNorm\n(B=1024, L=1024, d_model=8192)\nDevices 0-7', shape='parallelogram')
        stage0.node('layer0_mlp1', 'MLP Layer 1\n(B=1024, L=1024, d_model=8192) -> (B=1024, L=1024, 32768)\nTP across 8 devices\nEach: (B=1024, L=1024, 4096)')
        stage0.node('layer0_mlp2', 'MLP Layer 2\n(B=1024, L=1024, 32768) -> (B=1024, L=1024, d_model=8192)\nTP across 8 devices\nEach: (B=1024, L=1024, d_model=1024)')
        stage0.node('layer0_residual2', 'Residual Add\n(B=1024, L=1024, d_model=8192)\nDevices 0-7', shape='ellipse')
        
        # Layer 1
        stage0.node('layer1_norm1', 'LayerNorm\n(B=1024, L=1024, d_model=8192)\nDevices 0-7', shape='parallelogram')
        stage0.node('layer1_q_proj', 'Q Projection\n(B=1024, L=1024, d_model=8192) -> (B=1024, L=1024, d_model=8192)\nTP across 8 devices\nEach: (B=1024, L=1024, d_model=1024)')
        stage0.node('layer1_k_proj', 'K Projection\n(B=1024, L=1024, d_model=8192) -> (B=1024, L=1024, d_model=8192)\nTP across 8 devices\nEach: (B=1024, L=1024, d_model=1024)')
        stage0.node('layer1_v_proj', 'V Projection\n(B=1024, L=1024, d_model=8192) -> (B=1024, L=1024, d_model=8192)\nTP across 8 devices\nEach: (B=1024, L=1024, d_model=1024)')
        
        stage0.node('layer1_mha', 'Multi-Head Attention\nTP across 8 devices\nEach head: 2 heads/device\nLocal: (B=1024, L=1024, d_model=1024)')
        stage0.node('layer1_residual1', 'Residual Add\n(B=1024, L=1024, d_model=8192)\nDevices 0-7', shape='ellipse')
        
        stage0.node('layer1_norm2', 'LayerNorm\n(B=1024, L=1024, d_model=8192)\nDevices 0-7', shape='parallelogram')
        stage0.node('layer1_mlp1', 'MLP Layer 1\n(B=1024, L=1024, d_model=8192) -> (B=1024, L=1024, 32768)\nTP across 8 devices\nEach: (B=1024, L=1024, 4096)')
        stage0.node('layer1_mlp2', 'MLP Layer 2\n(B=1024, L=1024, 32768) -> (B=1024, L=1024, d_model=8192)\nTP across 8 devices\nEach: (B=1024, L=1024, d_model=1024)')
        stage0.node('layer1_residual2', 'Residual Add\n(B=1024, L=1024, d_model=8192)\nDevices 0-7', shape='ellipse')
    
    # Pipeline communication between stages
    dot.node('pipeline_comm', 'Pipeline Communication\n(B=1024, L=1024, d_model=8192)\nDevices 7 -> 8', shape='ellipse', fillcolor='yellow')
    
    # Pipeline Stage 1 (Devices 8-15, Layers 2-3)
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1\nDevices 8-15', style='dashed', color='red')
        
        # Layer 2
        stage1.node('layer2_norm1', 'LayerNorm\n(B=1024, L=1024, d_model=8192)\nDevices 8-15', shape='parallelogram')
        stage1.node('layer2_q_proj', 'Q Projection\n(B=1024, L=1024, d_model=8192) -> (B=1024, L=1024, d_model=8192)\nTP across 8 devices\nEach: (B=1024, L=1024, d_model=1024)')
        stage1.node('layer2_k_proj', 'K Projection\n(B=1024, L=1024, d_model=8192) -> (B=1024, L=1024, d_model=8192)\nTP across 8 devices\nEach: (B=1024, L=1024, d_model=1024)')
        stage1.node('layer2_v_proj', 'V Projection\n(B=1024, L=1024, d_model=8192) -> (B=1024, L=1024, d_model=8192)\nTP across 8 devices\nEach: (B=1024, L=1024, d_model=1024)')
        
        stage1.node('layer2_mha', 'Multi-Head Attention\nTP across 8 devices\nEach head: 2 heads/device\nLocal: (B=1024, L=1024, d_model=1024)')
        stage1.node('layer2_residual1', 'Residual Add\n(B=1024, L=1024, d_model=8192)\nDevices 8-15', shape='ellipse')
        
        stage1.node('layer2_norm2', 'LayerNorm\n(B=1024, L=1024, d_model=8192)\nDevices 8-15', shape='parallelogram')
        stage1.node('layer2_mlp1', 'MLP Layer 1\n(B=1024, L=1024, d_model=8192) -> (B=1024, L=1024, 32768)\nTP across 8 devices\nEach: (B=1024, L=1024, 4096)')
        stage1.node('layer2_mlp2', 'MLP Layer 2\n(B=1024, L=1024, 32768) -> (B=1024, L=1024, d_model=8192)\nTP across 8 devices\nEach: (B=1024, L=1024, d_model=1024)')
        stage1.node('layer2_residual2', 'Residual Add\n(B=1024, L=1024, d_model=8192)\nDevices 8-15', shape='ellipse')
        
        # Layer 3
        stage1.node('layer3_norm1', 'LayerNorm\n(B=1024, L=1024, d_model=8192)\nDevices 8-15', shape='parallelogram')
        stage1.node('layer3_q_proj', 'Q Projection\n(B=1024, L=1024, d_model=8192) -> (B=1024, L=1024, d_model=8192)\nTP across 8 devices\nEach: (B=1024, L=1024, d_model=1024)')
        stage1.node('layer3_k_proj', 'K Projection\n(B=1024, L=1024, d_model=8192) -> (B=1024, L=1024, d_model=8192)\nTP across 8 devices\nEach: (B=1024, L=1024, d_model=1024)')
        stage1.node('layer3_v_proj', 'V Projection\n(B=1024, L=1024, d_model=8192) -> (B=1024, L=1024, d_model=8192)\nTP across 8 devices\nEach: (B=1024, L=1024, d_model=1024)')
        
        stage1.node('layer3_mha', 'Multi-Head Attention\nTP across 8 devices\nEach head: 2 heads/device\nLocal: (B=1024, L=1024, d_model=1024)')
        stage1.node('layer3_residual1', 'Residual Add\n(B=1024, L=1024, d_model=8192)\nDevices 8-15', shape='ellipse')
        
        stage1.node('layer3_norm2', 'LayerNorm\n(B=1024, L=1024, d_model=8192)\nDevices 8-15', shape='parallelogram')
        stage1.node('layer3_mlp1', 'MLP Layer 1\n(B=1024, L=1024, d_model=8192) -> (B=1024, L=1024, 32768)\nTP across 8 devices\nEach: (B=1024, L=1024, 4096)')
        stage1.node('layer3_mlp2', 'MLP Layer 2\n(B=1024, L=1024, 32768) -> (B=1024, L=1024, d_model=8192)\nTP across 8 devices\nEach: (B=1024, L=1024, d_model=1024)')
        stage1.node('layer3_residual2', 'Residual Add\n(B=1024, L=1024, d_model=8192)\nDevices 8-15', shape='ellipse')
    
    # Output node
    dot.node('output', 'Output\n(B=1024, L=1024, d_model=8192)\nAll GPUs', shape='ellipse', fillcolor='lightgreen')
    
    # Connections for Stage 0
    dot.edge('input', 'layer0_norm1')
    dot.edge('layer0_norm1', 'layer0_q_proj')
    dot.edge('layer0_norm1', 'layer0_k_proj')
    dot.edge('layer0_norm1', 'layer0_v_proj')
    dot.edge('layer0_q_proj', 'layer0_mha')
    dot.edge('layer0_k_proj', 'layer0_mha')
    dot.edge('layer0_v_proj', 'layer0_mha')
    dot.edge('input', 'layer0_residual1')
    dot.edge('layer0_mha', 'layer0_residual1')
    dot.edge('layer0_residual1', 'layer0_norm2')
    dot.edge('layer0_norm2', 'layer0_mlp1')
    dot.edge('layer0_mlp1', 'layer0_mlp2')
    dot.edge('layer0_residual1', 'layer0_residual2')
    dot.edge('layer0_mlp2', 'layer0_residual2')
    
    dot.edge('layer0_residual2', 'layer1_norm1')
    dot.edge('layer1_norm1', 'layer1_q_proj')
    dot.edge('layer1_norm1', 'layer1_k_proj')
    dot.edge('layer1_norm1', 'layer1_v_proj')
    dot.edge('layer1_q_proj', 'layer1_mha')
    dot.edge('layer1_k_proj', 'layer1_mha')
    dot.edge('layer1_v_proj', 'layer1_mha')
    dot.edge('layer0_residual2', 'layer1_residual1')
    dot.edge('layer1_mha', 'layer1_residual1')
    dot.edge('layer1_residual1', 'layer1_norm2')
    dot.edge('layer1_norm2', 'layer1_mlp1')
    dot.edge('layer1_mlp1', 'layer1_mlp2')
    dot.edge('layer1_residual1', 'layer1_residual2')
    dot.edge('layer1_mlp2', 'layer1_residual2')
    
    # Pipeline communication
    dot.edge('layer1_residual2', 'pipeline_comm')
    
    # Connections for Stage 1
    dot.edge('pipeline_comm', 'layer2_norm1')
    dot.edge('layer2_norm1', 'layer2_q_proj')
    dot.edge('layer2_norm1', 'layer2_k_proj')
    dot.edge('layer2_norm1', 'layer2_v_proj')
    dot.edge('layer2_q_proj', 'layer2_mha')
    dot.edge('layer2_k_proj', 'layer2_mha')
    dot.edge('layer2_v_proj', 'layer2_mha')
    dot.edge('pipeline_comm', 'layer2_residual1')
    dot.edge('layer2_mha', 'layer2_residual1')
    dot.edge('layer2_residual1', 'layer2_norm2')
    dot.edge('layer2_norm2', 'layer2_mlp1')
    dot.edge('layer2_mlp1', 'layer2_mlp2')
    dot.edge('layer2_residual1', 'layer2_residual2')
    dot.edge('layer2_mlp2', 'layer2_residual2')
    
    dot.edge('layer2_residual2', 'layer3_norm1')
    dot.edge('layer3_norm1', 'layer3_q_proj')
    dot.edge('layer3_norm1', 'layer3_k_proj')
    dot.edge('layer3_norm1', 'layer3_v_proj')
    dot.edge('layer3_q_proj', 'layer3_mha')
    dot.edge('layer3_k_proj', 'layer3_mha')
    dot.edge('layer3_v_proj', 'layer3_mha')
    dot.edge('layer2_residual2', 'layer3_residual1')
    dot.edge('layer3_mha', 'layer3_residual1')
    dot.edge('layer3_residual1', 'layer3_norm2')
    dot.edge('layer3_norm2', 'layer3_mlp1')
    dot.edge('layer3_mlp1', 'layer3_mlp2')
    dot.edge('layer3_residual1', 'layer3_residual2')
    dot.edge('layer3_mlp2', 'layer3_residual2')
    
    # Output
    dot.edge('layer3_residual2', 'output')
    
    return dot

if __name__ == "__main__":
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/2025-09-03-17-15-16/baseline_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-03-17-15-16/baseline_dag.dot')
    print("Baseline DAG generated successfully")