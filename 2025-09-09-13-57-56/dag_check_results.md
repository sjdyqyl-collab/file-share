# DAG Validation Results

## Baseline MoE DAG Analysis
- **Cycle Check**: No cycles detected ✓
- **Input Node Check**: All non-input nodes have at least one input ✓
- **Output Node Check**: All non-output nodes have at least one output ✓
- **Nodes with only in-degree**: {'output', 'stage_1_layer_2_attention_qkv'}
- **Nodes with only out-degree**: {'layer_3_final_layernorm', 'input'}

## Proposed MoE DAG Analysis
- **Cycle Check**: No cycles detected ✓
- **Input Node Check**: All non-input nodes have at least one input ✓
- **Output Node Check**: All non-output nodes have at least one output ✓
- **Nodes with only in-degree**: {'output', 'layer_3_attention_qkv', 'layer_2_attention_qkv'}
- **Nodes with only out-degree**: {'layer_3_final_layernorm', 'layer_2_final_layernorm', 'input'}

## Conclusion
Both DAGs are valid directed acyclic graphs with proper connectivity. No errors found.