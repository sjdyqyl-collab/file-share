#!/usr/bin/env python3
"""
Validation and Demonstration Script for Cache Miss Rate Model

This script demonstrates the analytical model with various cache configurations
and matrix sizes to validate the predictions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cache_miss_analysis_model import CacheMissModel
import matplotlib.pyplot as plt
import numpy as np

def validate_model_variations():
    """Validate the model with different cache configurations"""
    
    # Test configurations
    configs = [
        {'name': 'Small L1', 'C': 32*1024, 'b': 64, 'assoc': 4},   # 32KB L1
        {'name': 'Large L1', 'C': 64*1024, 'b': 64, 'assoc': 4},   # 64KB L1
        {'name': 'L2 Cache', 'C': 256*1024, 'b': 64, 'assoc': 8},  # 256KB L2
        {'name': 'L3 Cache', 'C': 2*1024*1024, 'b': 64, 'assoc': 16}  # 2MB L3
    ]
    
    # Matrix sizes to test
    matrix_sizes = [64, 128, 256, 512, 1024]
    
    results = []
    
    for config in configs:
        C, b, assoc = config['C'], config['b'], config['assoc']
        S = (C // b) // assoc
        model = CacheMissModel(C, S, b, assoc)
        
        config_results = []
        for size in matrix_sizes:
            M = K = N = size
            analysis = model.analyze_matrix_multiplication(M, K, N)
            
            config_results.append({
                'size': size,
                'miss_rate': analysis['miss_analysis']['overall_miss_rate'],
                'total_misses': analysis['miss_analysis']['total_misses'],
                'working_set_ratio': analysis['matrix_params']['working_set_ratio']
            })
        
        results.append({
            'config_name': config['name'],
            'cache_size': C,
            'results': config_results
        })
    
    return results

def demonstrate_tiling_optimization():
    """Demonstrate the impact of loop tiling on cache performance"""
    
    # Configuration
    cache_config = {'C': 32*1024, 'b': 64, 'assoc': 4}
    S = (cache_config['C'] // cache_config['b']) // cache_config['assoc']
    model = CacheMissModel(cache_config['C'], S, cache_config['b'], cache_config['assoc'])
    
    # Test matrix sizes
    sizes = [128, 256, 512, 1024]
    
    print("=== Tiling Optimization Analysis ===")
    print(f"Cache: {cache_config['C']/1024:.0f}KB, {cache_config['b']}B blocks, {cache_config['assoc']}-way")
    print()
    
    for size in sizes:
        M = K = N = size
        
        # Without tiling (naive)
        naive_result = model.analyze_matrix_multiplication(M, K, N)
        
        # With optimal tiling
        tile_sizes = model.calculate_optimal_tile_sizes(M, K, N)
        
        print(f"Matrix {size}x{size}x{size}:")
        print(f"  Naive miss rate: {naive_result['miss_analysis']['overall_miss_rate']:.4f}")
        print(f"  Optimal tile sizes: M={tile_sizes['optimal_tile_M']}, "
              f"K={tile_sizes['optimal_tile_K']}, N={tile_sizes['optimal_tile_N']}")
        print(f"  Cache utilization: {tile_sizes['cache_utilization']:.2f}")
        print()

def analyze_reuse_patterns():
    """Analyze reuse patterns for different matrices"""
    
    cache_config = {'C': 32*1024, 'b': 64, 'assoc': 4}
    S = (cache_config['C'] // cache_config['b']) // cache_config['assoc']
    model = CacheMissModel(cache_config['C'], S, cache_config['b'], cache_config['assoc'])
    
    M, K, N = 256, 256, 256
    reuse_model = model.generate_reuse_distance_model(M, K, N)
    
    print("=== Reuse Pattern Analysis ===")
    print(f"Matrix: {M}x{K}x{N}")
    print()
    
    for matrix, data in reuse_model['reuse_analysis'].items():
        print(f"{matrix}:")
        print(f"  Total elements: {data['total_elements']:,}")
        print(f"  Unique cache blocks: {data['unique_blocks']:,}")
        print(f"  Reuse factor: {data['reuse_factor']}")
        print(f"  Temporal reuse: {data['temporal_reuse']}")
        print(f"  Spatial reuse: {data['spatial_reuse']}")
        print()

def generate_miss_rate_curves():
    """Generate miss rate curves for different cache sizes"""
    
    # Fixed matrix size
    M = K = N = 512
    
    # Cache sizes to test (from 8KB to 1MB)
    cache_sizes = [8*1024, 16*1024, 32*1024, 64*1024, 128*1024, 256*1024, 512*1024, 1024*1024]
    block_size = 64
    associativity = 4
    
    miss_rates = []
    working_set_ratios = []
    
    for cache_size in cache_sizes:
        S = (cache_size // block_size) // associativity
        model = CacheMissModel(cache_size, S, block_size, associativity)
        result = model.analyze_matrix_multiplication(M, K, N)
        
        miss_rates.append(result['miss_analysis']['overall_miss_rate'])
        working_set_ratios.append(result['matrix_params']['working_set_ratio'])
    
    # Print results
    print("=== Miss Rate vs Cache Size ===")
    print(f"Matrix: {M}x{K}x{N}")
    print()
    print("Cache Size  Working Set Ratio  Miss Rate")
    print("----------  -----------------  ---------")
    
    for i, cache_size in enumerate(cache_sizes):
        print(f"{cache_size/1024:6.0f}KB    {working_set_ratios[i]:8.2f}x          {miss_rates[i]:8.4f}")
    
    return cache_sizes, miss_rates, working_set_ratios

def validate_associativity_impact():
    """Analyze impact of associativity on conflict misses"""
    
    cache_size = 32*1024
    block_size = 64
    M = K = N = 256
    
    associativities = [1, 2, 4, 8, 16]
    
    print("=== Associativity Impact Analysis ===")
    print(f"Cache: {cache_size/1024:.0f}KB, Matrix: {M}x{K}x{N}")
    print()
    print("Assoc  Conflict Misses  Total Misses  Miss Rate")
    print("-----  ---------------  ------------  ---------")
    
    for assoc in associativities:
        S = (cache_size // block_size) // assoc
        model = CacheMissModel(cache_size, S, block_size, assoc)
        result = model.analyze_matrix_multiplication(M, K, N)
        
        conflict_misses = result['miss_analysis']['conflict_misses']
        total_misses = result['miss_analysis']['total_misses']
        miss_rate = result['miss_analysis']['overall_miss_rate']
        
        print(f"{assoc:5d}  {conflict_misses:14,}  {total_misses:12,}  {miss_rate:9.4f}")

def main():
    """Run all validation tests"""
    
    print("Cache Miss Rate Model Validation")
    print("=" * 50)
    print()
    
    # 1. Validate with different cache configurations
    print("1. Validating with different cache configurations...")
    results = validate_model_variations()
    
    # 2. Demonstrate tiling optimization
    print("2. Demonstrating tiling optimization...")
    demonstrate_tiling_optimization()
    
    # 3. Analyze reuse patterns
    print("3. Analyzing reuse patterns...")
    analyze_reuse_patterns()
    
    # 4. Generate miss rate curves
    print("4. Generating miss rate curves...")
    generate_miss_rate_curves()
    
    # 5. Validate associativity impact
    print("5. Validating associativity impact...")
    validate_associativity_impact()

if __name__ == "__main__":
    main()