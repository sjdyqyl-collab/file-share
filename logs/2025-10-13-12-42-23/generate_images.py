import graphviz
import os

# Set the working directory
work_dir = "/home/wzc/data/file-share/logs/2025-10-13-12-42-23"
os.chdir(work_dir)

# Generate SVG for full attention DAG
with open('full_attention_dag.dot', 'r') as f:
    full_dot_content = f.read()

full_graph = graphviz.Source(full_dot_content)
full_graph.format = 'svg'
full_graph.render('full_attention_dag', cleanup=True)

# Generate SVG for compact attention DAG (fixed)
with open('compact_attention_dag_fixed.dot', 'r') as f:
    compact_dot_content = f.read()

compact_graph = graphviz.Source(compact_dot_content)
compact_graph.format = 'svg'
compact_graph.render('compact_attention_dag_fixed', cleanup=True)

# Also rename the fixed dot file to replace the old one
import shutil
shutil.move('compact_attention_dag_fixed.dot', 'compact_attention_dag.dot')

print("SVG images generated successfully!")
print("Files created:")
print("- full_attention_dag.svg")
print("- compact_attention_dag.svg")
print("- compact_attention_dag.dot (fixed)")