# DAG Error Report

## Issue Found in compact_attention_dag.dot

**Error Type**: Node with only input but no output

**Problematic Node**: `mask_global`

**Description**: The `mask_global` node receives input from `frame_group` but has no outgoing edges to any other node. This violates the requirement that all nodes except the output node must have at least one output connection.

**Suggested Fix**: Connect `mask_global` to a subsequent node in the computation flow, likely to `dual_window` along with the other mask nodes.

## full_attention_dag.dot Status

No issues found - all validation criteria are satisfied.

## Summary
- full_attention_dag: ✓ Valid
- compact_attention_dag: ❌ Invalid (mask_global needs output connection)