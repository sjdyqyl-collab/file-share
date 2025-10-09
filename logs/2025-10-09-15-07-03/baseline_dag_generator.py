import graphviz

def create_baseline_dag():
    dot = graphviz.Digraph('baseline_model', comment='Baseline TP=8, PP=2 DAG')
    
    # Set graph attributes
    dot.attr(rankdir='TB', size='20,30')
    dot.attr('node', shape='rectangle', style='filled')
    
    # Layer 0: Input
    dot.node('input', 'Input\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: All GPUs', shape='ellipse', fillcolor='lightblue')
    
    # Layer 1: Embedding (PP stage 0)
    dot.node('embed', 'Embedding\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', fillcolor='lightgreen')
    
    # Layer 2: LayerNorm
    dot.node('ln1', 'LayerNorm\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', fillcolor='lightyellow')
    
    # Layer 3: Multi-Head Attention - split across 8 GPUs (TP=8)
    dot.node('q_proj', 'Q Projection\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nGPU: 0-7 (TP=8)', fillcolor='lightcoral')
    dot.node('k_proj', 'K Projection\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nGPU: 0-7 (TP=8)', fillcolor='lightcoral')
    dot.node('v_proj', 'V Projection\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nGPU: 0-7 (TP=8)', fillcolor='lightcoral')
    
    # Layer 4: Flash Attention (distributed across 8 GPUs)
    dot.node('flash_attn', 'Flash Attention\nInput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nOutput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nGPU: 0-7 (TP=8)', fillcolor='orange')
    
    # Layer 5: Output projection (TP=8)
    dot.node('o_proj', 'O Projection\nInput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7 (TP=8)', fillcolor='lightcoral')
    
    # Layer 6: Residual connection
    dot.node('res1', 'Residual Add\nInput: [batch_size=B, seq_len=S, hidden_size=4096], [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', fillcolor='lightgray')
    
    # Layer 7: LayerNorm
    dot.node('ln2', 'LayerNorm\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', fillcolor='lightyellow')
    
    # Layer 8: MLP - First Linear (Column Parallel)
    dot.node('mlp1', 'MLP First Linear\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, ffn_hidden_size=16384]\nGPU: 0-7 (TP=8)', fillcolor='lightpink')
    
    # Layer 9: GELU Activation
    dot.node('gelu', 'GELU Activation\nInput: [batch_size=B, seq_len=S, ffn_hidden_size=16384]\nOutput: [batch_size=B, seq_len=S, ffn_hidden_size=16384]\nGPU: 0-7', fillcolor='lightcyan')
    
    # Layer 10: MLP - Second Linear (Row Parallel)
    dot.node('mlp2', 'MLP Second Linear\nInput: [batch_size=B, seq_len=S, ffn_hidden_size=16384]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7 (TP=8)', fillcolor='lightpink')
    
    # Layer 11: Residual connection
    dot.node('res2', 'Residual Add\nInput: [batch_size=B, seq_len=S, hidden_size=4096], [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', fillcolor='lightgray')
    
    # Communication nodes
    dot.node('comm1', 'All-Reduce\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 0-7', shape='parallelogram', fillcolor='lightsteelblue')
    
    # Pipeline stage boundary
    dot.node('pipeline_comm', 'Pipeline Communication\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 7→8', shape='parallelogram', fillcolor='lightsteelblue')
    
    # Layer 12: Repeat for layers 2-4 (PP stage 1)
    dot.node('ln3', 'LayerNorm (Layer 2)\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 8-15', fillcolor='lightyellow')
    
    dot.node('q_proj2', 'Q Projection (Layer 2)\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nGPU: 8-15 (TP=8)', fillcolor='lightcoral')
    dot.node('k_proj2', 'K Projection (Layer 2)\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nGPU: 8-15 (TP=8)', fillcolor='lightcoral')
    dot.node('v_proj2', 'V Projection (Layer 2)\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nGPU: 8-15 (TP=8)', fillcolor='lightcoral')
    
    dot.node('flash_attn2', 'Flash Attention (Layer 2)\nInput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nOutput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nGPU: 8-15 (TP=8)', fillcolor='orange')
    dot.node('o_proj2', 'O Projection (Layer 2)\nInput: [batch_size=B, seq_len=S, heads=32, d_k=128]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 8-15 (TP=8)', fillcolor='lightcoral')
    dot.node('res3', 'Residual Add (Layer 2)\nInput: [batch_size=B, seq_len=S, hidden_size=4096], [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 8-15', fillcolor='lightgray')
    
    dot.node('ln4', 'LayerNorm (Layer 2)\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 8-15', fillcolor='lightyellow')
    dot.node('mlp3', 'MLP First Linear (Layer 2)\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, ffn_hidden_size=16384]\nGPU: 8-15 (TP=8)', fillcolor='lightpink')
    dot.node('gelu2', 'GELU Activation (Layer 2)\nInput: [batch_size=B, seq_len=S, ffn_hidden_size=16384]\nOutput: [batch_size=B, seq_len=S, ffn_hidden_size=16384]\nGPU: 8-15', fillcolor='lightcyan')
    dot.node('mlp4', 'MLP Second Linear (Layer 2)\nInput: [batch_size=B, seq_len=S, ffn_hidden_size=16384]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 8-15 (TP=8)', fillcolor='lightpink')
    dot.node('res4', 'Residual Add (Layer 2)\nInput: [batch_size=B, seq_len=S, hidden_size=4096], [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, hidden_size=4096]\nGPU: 8-15', fillcolor='lightgray')
    
    # Continue for layers 3-4 (simplified representation)
    dot.node('layer3', 'Layers 3-4\n[Same structure as above]\nGPU: 8-15', fillcolor='lightgreen')
    
    # Final output
    dot.node('output', 'Output\nInput: [batch_size=B, seq_len=S, hidden_size=4096]\nOutput: [batch_size=B, seq_len=S, vocab_size=V]\nGPU: 8-15', shape='ellipse', fillcolor='lightblue')
    
    # Create edges
    dot.edge('input', 'embed')
    dot.edge('embed', 'ln1')
    dot.edge('ln1', 'q_proj')
    dot.edge('ln1', 'k_proj')
    dot.edge('ln1', 'v_proj')
    dot.edge('q_proj', 'flash_attn')
    dot.edge('k_proj', 'flash_attn')
    dot.edge('v_proj', 'flash_attn')
    dot.edge('flash_attn', 'o_proj')
    dot.edge('o_proj', 'res1')
    dot.edge('ln1', 'res1')  # Residual connection
    dot.edge('res1', 'ln2')
    dot.edge('ln2', 'mlp1')
    dot.edge('mlp1', 'gelu')
    dot.edge('gelu', 'mlp2')
    dot.edge('mlp2', 'res2')
    dot.edge('res1', 'res2')  # Residual connection
    dot.edge('res2', 'comm1')
    dot.edge('comm1', 'pipeline_comm')
    dot.edge('pipeline_comm', 'ln3')
    dot.edge('ln3', 'q_proj2')
    dot.edge('ln3', 'k_proj2')
    dot.edge('ln3', 'v_proj2')
    dot.edge('q_proj2', 'flash_attn2')
    dot.edge('k_proj2', 'flash_attn2')
    dot.edge('v_proj2', 'flash_attn2')
    dot.edge('flash_attn2', 'o_proj2')
    dot.edge('o_proj2', 'res3')
    dot.edge('ln3', 'res3')  # Residual connection
    dot.edge('res3', 'ln4')
    dot.edge('ln4', 'mlp3')
    dot.edge('mlp3', 'gelu2')
    dot.edge('gelu2', 'mlp4')
    dot.edge('mlp4', 'res4')
    dot.edge('res3', 'res4')  # Residual connection
    dot.edge('res4', 'layer3')
    dot.edge('layer3', 'output')
    
    return dot

if __name__ == "__main__":
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/logs/2025-10-09-15-07-03/baseline_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/logs/2025-10-09-15-07-03/baseline_dag.dot')
    print("Baseline DAG generated successfully")