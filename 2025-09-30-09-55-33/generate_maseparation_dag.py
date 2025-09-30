#!/usr/bin/env python3
"""
Generate DAGs for MA Separation paper
"""

import graphviz
from typing import Dict, List, Tuple

def create_maseparation_dag():
    """Create MA Separation DAG with 16 GPUs"""
    dot = graphviz.Digraph('MA_Separation', comment='MA Separation Architecture')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
    
    # Define GPU clusters
    with dot.subgraph(name='cluster_attention') as att:
        att.attr(label='Attention GPUs (0-7)', style='rounded', bgcolor='lightblue')
        
        # Input layer
        att.node('input', 'Input Layer\n[batch_size=1024, seq_len=2048, hidden_dim=4096]', 
                shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        # Layer 1: QKV Projection (split across 8 GPUs)
        for i in range(8):
            att.node(f'qkv_proj_{i}', f'QKV Projection GPU{i}\n[1024,2048,512]\n4 heads×128', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Layer 2: Attention Computation
        for i in range(8):
            att.node(f'attn_{i}', f'Attention GPU{i}\n[1024,2048,512]\n4 heads', 
                    shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Layer 3: Output Aggregation
        att.node('attn_agg', 'Attention All-Reduce\n[1024,2048,4096]\n8×512→4096', 
                shape='ellipse', style='filled', fillcolor='orange')
        
        # Attention residual add
        att.node('attn_residual', 'Attention Residual Add\n[1024,2048,4096]', 
                shape='ellipse', style='filled', fillcolor='pink')
        
        # Layer normalization
        att.node('attn_ln', 'Layer Norm\n[1024,2048,4096]', 
                shape='rectangle', style='filled', fillcolor='lightblue')
    
    with dot.subgraph(name='cluster_moe') as moe:
        moe.attr(label='MoE GPUs (8-15)', style='rounded', bgcolor='lightgreen')
        
        # Gate computation
        moe.node('gate', 'Gate Network\n[1024×2048,4096]→[1024×2048,16]\nTop-2 routing', 
                shape='rectangle', style='filled, dashed', fillcolor='yellow')
        
        # Expert processing (2 experts per GPU)
        for gpu in range(8):
            for expert in range(2):
                expert_id = gpu * 2 + expert
                moe.node(f'expert_{expert_id}', f'Expert{expert_id} GPU{gpu+8}\n[1024×2048,4096]→[1024×2048,4096]\n16384 hidden', 
                        shape='rectangle', style='filled', fillcolor='lightcyan')
        
        # Expert aggregation
        moe.node('expert_agg', 'Expert Aggregation\n[1024×2048,4096]\nWeighted sum', 
                shape='ellipse', style='filled', fillcolor='orange')
        
        # MoE residual add
        moe.node('moe_residual', 'MoE Residual Add\n[1024,2048,4096]', 
                shape='ellipse', style='filled', fillcolor='pink')
        
        # Layer normalization
        moe.node('moe_ln', 'Layer Norm\n[1024,2048,4096]', 
                shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Output layer
    dot.node('output', 'Output Layer\n[batch_size=1024, seq_len=2048, hidden_dim=4096]', 
            shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Connect attention components
    dot.edge('input', 'qkv_proj_0')
    dot.edge('input', 'qkv_proj_1')
    dot.edge('input', 'qkv_proj_2')
    dot.edge('input', 'qkv_proj_3')
    dot.edge('input', 'qkv_proj_4')
    dot.edge('input', 'qkv_proj_5')
    dot.edge('input', 'qkv_proj_6')
    dot.edge('input', 'qkv_proj_7')
    
    for i in range(8):
        dot.edge(f'qkv_proj_{i}', f'attn_{i}')
        dot.edge(f'attn_{i}', 'attn_agg')
    
    dot.edge('attn_agg', 'attn_residual')
    dot.edge('input', 'attn_residual')  # Residual connection
    dot.edge('attn_residual', 'attn_ln')
    
    # Connect attention to MoE with communication
    dot.edge('attn_ln', 'gate', label='Cross-GPU\nCommunication', style='dashed')
    for expert_id in range(16):
        dot.edge('gate', f'expert_{expert_id}', style='dashed', label=f'route tokens')
    
    for expert_id in range(16):
        dot.edge(f'expert_{expert_id}', 'expert_agg')
    
    dot.edge('expert_agg', 'moe_residual')
    dot.edge('attn_ln', 'moe_residual')  # Residual connection
    dot.edge('moe_residual', 'moe_ln')
    dot.edge('moe_ln', 'output')
    
    return dot

def create_baseline_tp8_dag():
    """Create Tensor Parallelism (TP=8) baseline DAG"""
    dot = graphviz.Digraph('TP8_Baseline', comment='Tensor Parallelism 8-Way')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
    
    # Input
    dot.node('input', 'Input Layer\n[batch_size=1024, seq_len=2048, hidden_dim=4096]', 
            shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Model parallel split across 8 GPUs
    for gpu in range(8):
        with dot.subgraph(name=f'cluster_gpu{gpu}') as c:
            c.attr(label=f'GPU {gpu}', style='rounded', bgcolor='lightgray')
            
            # QKV projection (split by hidden dimension)
            c.node(f'qkv_{gpu}', f'QKV Projection\n[1024,2048,512]\nHidden split', 
                   shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # Attention (split by heads)
            c.node(f'attn_{gpu}', f'Multi-Head Attention\n[1024,2048,512]\n4 heads', 
                   shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # FFN/MoE (split by hidden dimension)
            c.node(f'moe_{gpu}', f'MoE Layer\n[1024,2048,512]\n2 experts', 
                   shape='rectangle', style='filled', fillcolor='lightcyan')
            
            # Layer norm
            c.node(f'ln_{gpu}', f'Layer Norm\n[1024,2048,512]', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
    
    # All-reduce operations
    dot.node('ar1', 'All-Reduce\nAttention Output\n[1024,2048,4096]', 
            shape='ellipse', style='filled', fillcolor='orange')
    dot.node('ar2', 'All-Reduce\nMoE Output\n[1024,2048,4096]', 
            shape='ellipse', style='filled', fillcolor='orange')
    
    # Output
    dot.node('output', 'Output Layer\n[batch_size=1024, seq_len=2048, hidden_dim=4096]', 
            shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Connections
    for gpu in range(8):
        dot.edge('input', f'qkv_{gpu}')
        dot.edge(f'qkv_{gpu}', f'attn_{gpu}')
        dot.edge(f'attn_{gpu}', f'ln_{gpu}')
        dot.edge(f'ln_{gpu}', f'moe_{gpu}')
        dot.edge(f'moe_{gpu}', f'ar1')
    
    dot.edge('ar1', 'ar2')
    dot.edge('ar2', 'output')
    
    return dot

def create_baseline_pp2_dag():
    """Create Pipeline Parallelism (PP=2) baseline DAG"""
    dot = graphviz.Digraph('PP2_Baseline', comment='Pipeline Parallelism 2-Way')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
    
    # Input
    dot.node('input', 'Input Layer\n[batch_size=1024, seq_len=2048, hidden_dim=4096]', 
            shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Stage 0: Layers 0-1
    with dot.subgraph(name='cluster_stage0') as s0:
        s0.attr(label='Stage 0 (GPUs 0-7)\nLayers 0-1', style='rounded', bgcolor='lightblue')
        
        for gpu in range(8):
            s0.node(f's0_l0_gpu{gpu}', f'Layer0 GPU{gpu}\n[1024,2048,4096]', 
                   shape='rectangle', style='filled', fillcolor='lightgreen')
            s0.node(f's0_l1_gpu{gpu}', f'Layer1 GPU{gpu}\n[1024,2048,4096]', 
                   shape='rectangle', style='filled', fillcolor='lightcoral')
    
    # Stage 1: Layers 2-3
    with dot.subgraph(name='cluster_stage1') as s1:
        s1.attr(label='Stage 1 (GPUs 8-15)\nLayers 2-3', style='rounded', bgcolor='lightgreen')
        
        for gpu in range(8, 16):
            s1.node(f's1_l2_gpu{gpu}', f'Layer2 GPU{gpu}\n[1024,2048,4096]', 
                   shape='rectangle', style='filled', fillcolor='lightcyan')
            s1.node(f's1_l3_gpu{gpu}', f'Layer3 GPU{gpu}\n[1024,2048,4096]', 
                   shape='rectangle', style='filled', fillcolor='lightyellow')
    
    # Pipeline communication
    dot.node('pipe_comm', 'Pipeline Communication\nMicro-batch transfer', 
            shape='ellipse', style='filled', fillcolor='orange')
    
    # Output
    dot.node('output', 'Output Layer\n[batch_size=1024, seq_len=2048, hidden_dim=4096]', 
            shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Connections
    for gpu in range(8):
        dot.edge('input', f's0_l0_gpu{gpu}')
        dot.edge(f's0_l0_gpu{gpu}', f's0_l1_gpu{gpu}')
        dot.edge(f's0_l1_gpu{gpu}', 'pipe_comm')
    
    for gpu in range(8, 16):
        dot.edge('pipe_comm', f's1_l2_gpu{gpu}')
        dot.edge(f's1_l2_gpu{gpu}', f's1_l3_gpu{gpu}')
        dot.edge(f's1_l3_gpu{gpu}', 'output')
    
    return dot

def create_baseline_tppp_dag():
    """Create Hybrid TP+PP (TP=8, PP=2) baseline DAG"""
    dot = graphviz.Digraph('TPPP_Baseline', comment='Hybrid TP=8, PP=2')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
    
    # Input
    dot.node('input', 'Input Layer\n[batch_size=1024, seq_len=2048, hidden_dim=4096]', 
            shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Stage 0: Layers 0-1 with TP=8
    with dot.subgraph(name='cluster_stage0') as s0:
        s0.attr(label='Stage 0 (GPUs 0-7)\nTP=8, Layers 0-1', style='rounded', bgcolor='lightblue')
        
        for gpu in range(8):
            s0.node(f's0_l0_tp{gpu}', f'Layer0 GPU{gpu}\n[1024,2048,512]\nTP split', 
                   shape='rectangle', style='filled', fillcolor='lightgreen')
            s0.node(f's0_l1_tp{gpu}', f'Layer1 GPU{gpu}\n[1024,2048,512]\nTP split', 
                   shape='rectangle', style='filled', fillcolor='lightcoral')
            s0.node(f's0_ar0_{gpu}', f'All-Reduce\nLayer0\n[1024,2048,4096]', 
                   shape='ellipse', style='filled', fillcolor='orange')
            s0.node(f's0_ar1_{gpu}', f'All-Reduce\nLayer1\n[1024,2048,4096]', 
                   shape='ellipse', style='filled', fillcolor='orange')
    
    # Stage 1: Layers 2-3 with TP=8
    with dot.subgraph(name='cluster_stage1') as s1:
        s1.attr(label='Stage 1 (GPUs 8-15)\nTP=8, Layers 2-3', style='rounded', bgcolor='lightgreen')
        
        for gpu in range(8, 16):
            gpu_idx = gpu - 8
            s1.node(f's1_l2_tp{gpu}', f'Layer2 GPU{gpu}\n[1024,2048,512]\nTP split', 
                   shape='rectangle', style='filled', fillcolor='lightcyan')
            s1.node(f's1_l3_tp{gpu}', f'Layer3 GPU{gpu}\n[1024,2048,512]\nTP split', 
                   shape='rectangle', style='filled', fillcolor='lightyellow')
            s1.node(f's1_ar2_{gpu}', f'All-Reduce\nLayer2\n[1024,2048,4096]', 
                   shape='ellipse', style='filled', fillcolor='orange')
            s1.node(f's1_ar3_{gpu}', f'All-Reduce\nLayer3\n[1024,2048,4096]', 
                   shape='ellipse', style='filled', fillcolor='orange')
    
    # Pipeline communication
    dot.node('pipe_comm', 'Pipeline Communication\nMicro-batch transfer', 
            shape='ellipse', style='filled', fillcolor='orange')
    
    # Output
    dot.node('output', 'Output Layer\n[batch_size=1024, seq_len=2048, hidden_dim=4096]', 
            shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Connections
    for gpu in range(8):
        dot.edge('input', f's0_l0_tp{gpu}')
        dot.edge(f's0_l0_tp{gpu}', f's0_ar0_{gpu}')
        dot.edge(f's0_ar0_{gpu}', f's0_l1_tp{gpu}')
        dot.edge(f's0_l1_tp{gpu}', f's0_ar1_{gpu}')
        dot.edge(f's0_ar1_{gpu}', 'pipe_comm')
    
    for gpu in range(8, 16):
        gpu_idx = gpu - 8
        dot.edge('pipe_comm', f's1_l2_tp{gpu}')
        dot.edge(f's1_l2_tp{gpu}', f's1_ar2_{gpu}')
        dot.edge(f's1_ar2_{gpu}', f's1_l3_tp{gpu}')
        dot.edge(f's1_l3_tp{gpu}', f's1_ar3_{gpu}')
        dot.edge(f's1_ar3_{gpu}', 'output')
    
    return dot

if __name__ == '__main__':
    # Generate all DAGs
    print("Generating MA Separation DAG...")
    masep_dag = create_maseparation_dag()
    masep_dag.render('/home/wzc/data/file-share/2025-09-30-09-55-33/ma_separation', format='svg', cleanup=False)
    masep_dag.save('/home/wzc/data/file-share/2025-09-30-09-55-33/ma_separation.dot')
    
    print("Generating TP=8 Baseline DAG...")
    tp8_dag = create_baseline_tp8_dag()
    tp8_dag.render('/home/wzc/data/file-share/2025-09-30-09-55-33/baseline_tp8', format='svg', cleanup=False)
    tp8_dag.save('/home/wzc/data/file-share/2025-09-30-09-55-33/baseline_tp8.dot')
    
    print("Generating PP=2 Baseline DAG...")
    pp2_dag = create_baseline_pp2_dag()
    pp2_dag.render('/home/wzc/data/file-share/2025-09-30-09-55-33/baseline_pp2', format='svg', cleanup=False)
    pp2_dag.save('/home/wzc/data/file-share/2025-09-30-09-55-33/baseline_pp2.dot')
    
    print("Generating TP+PP Baseline DAG...")
    tppp_dag = create_baseline_tppp_dag()
    tppp_dag.render('/home/wzc/data/file-share/2025-09-30-09-55-33/baseline_tppp', format='svg', cleanup=False)
    tppp_dag.save('/home/wzc/data/file-share/2025-09-30-09-55-33/baseline_tppp.dot')
    
    print("All DAGs generated successfully!")