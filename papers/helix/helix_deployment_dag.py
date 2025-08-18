import graphviz

# Create a new directed graph
dot = graphviz.Digraph(comment='Helix Two-Level Attention Partitioning on 16 GPUs')
dot.attr(rankdir='TB', size='30,20')
dot.attr('node', shape='box', style='filled')

# Define colors for different GPU groups
colors = [
    '#FF9999', '#FFCC99', '#FFFF99', '#99FF99',  # GPU 0-3 (Head Group 0)
    '#99CCFF', '#CC99FF', '#FF99CC', '#FFCCFF',  # GPU 4-7 (Head Group 1)
    '#FFB366', '#B3FF66', '#66B3FF', '#B366FF',  # GPU 8-11 (Head Group 2)
    '#FF6666', '#66FF66', '#6666FF', '#FF66FF'   # GPU 12-15 (Head Group 3)
]

# Input layer
dot.node('input', 'Input\nX: [B, L, D]', fillcolor='lightgray')

# Define the 16 partitions (4 head groups × 4 dimension slices)
partitions = []
for i in range(4):  # head groups
    for j in range(4):  # dimension slices
        gpu_id = i * 4 + j
        node_id = f'gpu_{gpu_id}'
        label = f'GPU {gpu_id}\nHead Group {i}\nDim Slice {j}\nQ/K/V Proj: [D, d_s*h_g]\nAttention: [B, L, d_s*h_g]'
        dot.node(node_id, label, fillcolor=colors[gpu_id])
        partitions.append(node_id)

# Input projections - each GPU gets full input but different weight matrices
for gpu_id in range(16):
    dot.edge('input', f'gpu_{gpu_id}', label='X: [B, L, D]')

# Define communication patterns for aggregation
# First level: concatenate dimension slices within each head group
concat_nodes = []
for i in range(4):  # head groups
    concat_id = f'concat_group_{i}'
    dot.node(concat_id, f'Concat Group {i}\n[B, L, d*h_g]', fillcolor='lightblue')
    concat_nodes.append(concat_id)
    
    # Connect GPUs in the same head group to their concat node
    for j in range(4):  # dimension slices
        gpu_id = i * 4 + j
        dot.edge(f'gpu_{gpu_id}', concat_id, 
                label=f'Attention^{i,j}: [B, L, d_s*h_g]')

# Second level: concatenate head groups
final_concat = 'final_output'
dot.node(final_concat, 'Final Output\n[B, L, D]', fillcolor='lightgreen')

for i in range(4):
    dot.edge(f'concat_group_{i}', final_concat, 
            label=f'Group {i} Output: [B, L, d*h_g]')

# Add communication edges showing data flow between GPUs
dot.attr('edge', style='dashed', color='red')

# Show AllGather operations for dimension concatenation
for i in range(4):  # head groups
    for j in range(4):  # dimension slices
        gpu_id = i * 4 + j
        # Show communication within head group for dimension concatenation
        for k in range(4):
            if j != k:
                other_gpu = i * 4 + k
                dot.edge(f'gpu_{gpu_id}', f'gpu_{other_gpu}', 
                        label=f'AllGather Dim Slice', constraint='false')

# Show communication for head group concatenation
dot.attr('edge', style='dotted', color='blue')
for i in range(4):
    for j in range(4):
        if i != j:
            dot.edge(f'concat_group_{i}', f'concat_group_{j}', 
                    label='Concat Head Groups', constraint='false')

# Add subgraphs for visual grouping
with dot.subgraph(name='cluster_0') as c:
    c.attr(label='Head Group 0 (GPUs 0-3)', style='dashed')
    for j in range(4):
        c.node(f'gpu_{j}')

with dot.subgraph(name='cluster_1') as c:
    c.attr(label='Head Group 1 (GPUs 4-7)', style='dashed')
    for j in range(4):
        c.node(f'gpu_{4+j}')

with dot.subgraph(name='cluster_2') as c:
    c.attr(label='Head Group 2 (GPUs 8-11)', style='dashed')
    for j in range(4):
        c.node(f'gpu_{8+j}')

with dot.subgraph(name='cluster_3') as c:
    c.attr(label='Head Group 3 (GPUs 12-15)', style='dashed')
    for j in range(4):
        c.node(f'gpu_{12+j}')

# Save the DAG
dot.render('/home/wzc/data/papers/helix/helix_deployment_dag', format='dot', cleanup=False)

# Also save as plain text for inspection
with open('/home/wzc/data/papers/helix/helix_deployment_dag.gv', 'w') as f:
    f.write(dot.source)

print("DAG generated successfully!")
print("Files created:")
print("- helix_deployment_dag.gv (Graphviz source)")
print("- helix_deployment_dag.dot (compiled)")