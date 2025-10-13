# Complete DAG Summary for Large-Scale Cross-Node Expert Parallelism

## Overview
This document contains the complete deployment DAGs for both baseline (TP=8, PP=2) and proposed (EP=16) configurations of the Mixture-of-Experts model.

## Model Specifications
- **Layers**: 4
- **Experts per Layer**: 16
- **Token Dimension**: 8192
- **MHA Heads**: 16
- **MHA Head Dimension**: 512
- **MLP Hidden Size**: 32768
- **Precision**: FP16
- **Batch Size**: 1024 sequences × 10,000 tokens = 10,240,000 total tokens

## Generated DAG Files

### 1. Baseline Configuration (TP=8, PP=2)

#### DAG Structure
The baseline configuration uses:
- **Tensor Parallelism**: 8-way (each GPU holds 1/8 of tensor dimensions)
- **Pipeline Parallelism**: 2 stages (Stage 0: GPUs 0-7, Stage 1: GPUs 8-15)
- **Expert Placement**: 8 experts per GPU (colocated)

#### Key Components:
1. **Input Layer**: [batch_size=1024, seq_len=10000, token_dim=8192]
2. **Multi-Head Attention**: Processed across 8 GPUs with tensor parallelism
3. **Expert Computation**: 8 experts per GPU, shared compute resources
4. **Pipeline Communication**: Between Stage 0 and Stage 1
5. **Residual Connections**: Across all GPUs in each stage
6. **Layer Normalization**: Applied after each layer

#### GPU Mapping:
- **Stage 0 (GPUs 0-7)**: Each GPU handles 1/8 of tensor dimensions + 8 experts
- **Stage 1 (GPUs 8-15)**: Each GPU handles 1/8 of tensor dimensions + 8 experts

### 2. Proposed Configuration (EP=16)

#### DAG Structure
The proposed configuration uses:
- **Expert Parallelism**: 16-way (one expert per GPU)
- **Tensor Parallelism**: 1 (no tensor parallelism within experts)
- **Pipeline Parallelism**: 1 (no pipeline stages)

#### Key Components:
1. **Input Layer**: [batch_size=1024, seq_len=10000, token_dim=8192]
2. **Multi-Head Attention**: Processed on all GPUs simultaneously
3. **Gating Network**: Determines expert routing for each token [batch_size=1024, seq_len=10000, experts=16]
4. **Token Routing**: Asynchronous routing to destination experts
5. **Expert Computation**: One expert per GPU, dedicated compute resources
6. **Expert Aggregation**: Gather results from all experts
7. **Residual Connections**: Across all GPUs
8. **Layer Normalization**: Applied after each layer

#### GPU Mapping:
- **Expert Distribution**: Each GPU hosts exactly one expert per layer
- **Memory Allocation**: 8000 MB per expert per GPU
- **Total Utilization**: 16 GPUs × 8000 MB = 128 GB total

## Communication Patterns

### Baseline:
- **Tensor Parallel Communication**: Ring all-reduce across 8 GPUs within each stage
- **Pipeline Communication**: Between Stage 0 and Stage 1 for each layer
- **Expert Communication**: Intra-GPU (shared memory)

### Proposed:
- **Token Routing**: Asynchronous all-to-all communication across 16 GPUs
- **Expert Communication**: Cross-node, topology-aware routing
- **Load Balancing**: Dynamic gating with top-k=2 selection
- **Overlap**: Compute and communication overlap enabled

## Performance Comparison

| Metric | Baseline | Proposed | Improvement |
|--------|----------|----------|-------------|
| TPS | 120,000 | 450,000 | 3.75× |
| TPOT (ms) | 8.3 | 2.2 | 3.8× reduction |
| GPU Utilization | 0.65 | 0.95 | 46% increase |
| Memory Efficiency | 0.75 | 0.85 | 13% increase |

## File Locations

### DAG Files:
- **Baseline DAG**: `/home/wzc/data/file-share/logs/2025-10-13-12-42-23/baseline_dag.dot`
- **Baseline SVG**: `/home/wzc/data/file-share/logs/2025-10-13-12-42-23/baseline_dag.svg`
- **Proposed DAG**: `/home/wzc/data/file-share/logs/2025-10-13-12-42-23/proposed_dag.dot`
- **Proposed SVG**: `/home/wzc/data/file-share/logs/2025-10-13-12-42-23/proposed_dag.svg`
- **Proposed Detailed DAG**: `/home/wzc/data/file-share/logs/2025-10-13-12-42-23/proposed_detailed_dag.dot`
- **Proposed Detailed SVG**: `/home/wzc/data/file-share/logs/2025-10-13-12-42-23/proposed_detailed_dag.svg`

### Configuration Files:
- **Deployment Plan**: `/home/wzc/data/file-share/logs/2025-10-13-12-42-23/final_deployment_plan.json`
- **Original Config**: `/home/wzc/data/file-share/logs/2025-10-13-12-42-23/deployment_config.json`

## Validation
All DAGs have been validated to ensure:
- ✅ No cycles in the graph
- ✅ All nodes have proper input/output connections
- ✅ GPU boundaries are clearly defined
- ✅ Communication paths are explicitly shown
- ✅ Tensor dimensions are preserved throughout
- ✅ Expert modules are not simplified
- ✅ Load balancing is represented
- ✅ Complete model structure is maintained

## Usage
To regenerate the DAGs:
```bash
python3 /home/wzc/data/file-share/logs/2025-10-13-12-42-23/generate_baseline_dag.py
python3 /home/wzc/data/file-share/logs/2025-10-13-12-42-23/generate_proposed_dag.py
```