#!/usr/bin/env python3
"""
Analytical Cache Miss Rate Model for Matrix Multiplication

Cache Parameters:
- Cache Size: C bytes
- Set Size: S sets
- Block Size: b bytes
- Set Associativity: 4-way
- Replacement Policy: LRU
- Application: Matrix Multiplication MxKxN

The model analyzes cache miss rates for the standard triple-loop matrix multiplication:
C = A × B, where:
- A is M×K matrix
- B is K×N matrix  
- C is M×N matrix
"""

import math
from typing import Dict, List, Tuple

class CacheMissModel:
    def __init__(self, cache_size: int, num_sets: int, block_size: int, 
                 associativity: int = 4):
        """
        Initialize cache parameters
        
        Args:
            cache_size: Total cache size in bytes
            num_sets: Number of cache sets
            block_size: Block size in bytes
            associativity: Set associativity (default 4)
        """
        self.C = cache_size
        self.S = num_sets
        self.b = block_size
        self.assoc = associativity
        
        # Derived parameters
        self.blocks_per_set = associativity
        self.total_blocks = cache_size // block_size
        self.sets = num_sets
        self.blocks_per_set = self.total_blocks // num_sets
        
        # Validate parameters
        assert self.total_blocks == num_sets * associativity, \
            "Cache parameters inconsistent"
    
    def analyze_matrix_multiplication(self, M: int, K: int, N: int, 
                                    element_size: int = 8) -> Dict:
        """
        Analyze cache miss rates for matrix multiplication C = A×B
        
        Args:
            M: Rows in A and C
            K: Columns in A, rows in B
            N: Columns in B and C
            element_size: Size of each element in bytes (default 8 for double)
            
        Returns:
            Dictionary with miss rate analysis
        """
        
        # Matrix sizes in bytes
        A_size = M * K * element_size
        B_size = K * N * element_size
        C_size = M * N * element_size
        
        # Elements per cache block
        elements_per_block = self.b // element_size
        
        # Cache capacity analysis
        total_data_size = A_size + B_size + C_size
        working_set_ratio = total_data_size / self.C
        
        # Initialize results
        results = {
            'cache_params': {
                'cache_size_bytes': self.C,
                'num_sets': self.S,
                'block_size': self.b,
                'associativity': self.assoc,
                'total_blocks': self.total_blocks
            },
            'matrix_params': {
                'M': M, 'K': K, 'N': N,
                'element_size': element_size,
                'A_size_bytes': A_size,
                'B_size_bytes': B_size,
                'C_size_bytes': C_size,
                'total_data_bytes': total_data_size,
                'working_set_ratio': working_set_ratio
            },
            'miss_analysis': {}
        }
        
        # Analyze each type of miss
        results['miss_analysis']['capacity_misses'] = self._analyze_capacity_misses(M, K, N, element_size)
        results['miss_analysis']['conflict_misses'] = self._analyze_conflict_misses(M, K, N, element_size)
        results['miss_analysis']['compulsory_misses'] = self._analyze_compulsory_misses(M, K, N, element_size)
        results['miss_analysis']['total_misses'] = self._calculate_total_misses(results['miss_analysis'])
        results['miss_analysis']['total_accesses'] = self._calculate_total_accesses(M, K, N)
        results['miss_analysis']['overall_miss_rate'] = results['miss_analysis']['total_misses'] / results['miss_analysis']['total_accesses']
        
        return results
    
    def _analyze_capacity_misses(self, M: int, K: int, N: int, element_size: int) -> int:
        """
        Analyze capacity misses for matrix multiplication
        
        Capacity misses occur when the working set exceeds cache capacity
        """
        # For matrix multiplication, we need to consider the reuse patterns
        # Each element of A is reused N times
        # Each element of B is reused M times
        # Each element of C is reused K times
        
        # Effective cache size considering reuse
        effective_cache_blocks = self.total_blocks
        
        # Calculate if matrices fit in cache
        A_blocks = (M * K * element_size + self.b - 1) // self.b
        B_blocks = (K * N * element_size + self.b - 1) // self.b
        C_blocks = (M * N * element_size + self.b - 1) // self.b
        
        total_matrix_blocks = A_blocks + B_blocks + C_blocks
        
        if total_matrix_blocks <= effective_cache_blocks:
            # All data fits, no capacity misses beyond compulsory
            return 0
        
        # Calculate capacity misses based on working set size
        # This is a simplified model - in practice, depends on access pattern
        cache_lines_per_set = self.assoc
        total_sets = self.S
        
        # Estimate capacity misses based on LRU behavior
        # For large matrices, capacity misses dominate
        capacity_misses = 0
        
        # Each iteration of the inner loop accesses:
        # - One element from A (row i, column k)
        # - One element from B (row k, column j)
        # - One element from C (row i, column j)
        
        # For large matrices, we model capacity misses as:
        # When the working set for a given i,j iteration exceeds cache
        working_set_per_iteration = min(K, self.total_blocks // 3) * 3 * self.b
        
        if M * N * K * 3 * element_size > self.C:
            # Capacity misses occur when data is evicted and reloaded
            # Conservative estimate: each cache line is evicted and reloaded
            capacity_misses = max(0, total_matrix_blocks - effective_cache_blocks) * \
                            min(M, N, K)  # Approximate reuse factor
        
        return int(capacity_misses)
    
    def _analyze_conflict_misses(self, M: int, K: int, N: int, element_size: int) -> int:
        """
        Analyze conflict misses due to set associativity
        
        Conflict misses occur when multiple memory blocks map to the same set
        """
        # Calculate memory addresses for matrices
        # Assuming matrices are allocated contiguously
        A_start = 0
        B_start = A_start + M * K * element_size
        C_start = B_start + K * N * element_size
        
        # Calculate set indices for each matrix
        def get_set_index(addr: int) -> int:
            block_number = addr // self.b
            return block_number % self.S
        
        # Count accesses to each set
        set_access_counts = [0] * self.S
        
        # Simulate the matrix multiplication access pattern
        for i in range(M):
            for j in range(N):
                for k in range(K):
                    # Access A[i][k]
                    a_addr = A_start + (i * K + k) * element_size
                    set_idx = get_set_index(a_addr)
                    set_access_counts[set_idx] += 1
                    
                    # Access B[k][j]
                    b_addr = B_start + (k * N + j) * element_size
                    set_idx = get_set_index(b_addr)
                    set_access_counts[set_idx] += 1
                    
                    # Access C[i][j]
                    c_addr = C_start + (i * N + j) * element_size
                    set_idx = get_set_index(c_addr)
                    set_access_counts[set_idx] += 1
        
        # Calculate conflict misses
        conflict_misses = 0
        for count in set_access_counts:
            # If more than associativity unique blocks accessed to a set
            if count > self.assoc:
                # Each access beyond associativity causes a conflict miss
                # This is a simplified model
                conflict_misses += max(0, count - self.assoc)
        
        return int(conflict_misses)
    
    def _analyze_compulsory_misses(self, M: int, K: int, N: int, element_size: int) -> int:
        """
        Analyze compulsory (cold) misses
        
        Compulsory misses occur on first access to each cache block
        """
        # Calculate total unique cache blocks accessed
        A_blocks = (M * K * element_size + self.b - 1) // self.b
        B_blocks = (K * N * element_size + self.b - 1) // self.b
        C_blocks = (M * N * element_size + self.b - 1) // self.b
        
        # Each unique block causes at least one compulsory miss
        compulsory_misses = A_blocks + B_blocks + C_blocks
        
        return int(compulsory_misses)
    
    def _calculate_total_misses(self, miss_analysis: Dict) -> int:
        """Calculate total misses from different categories"""
        return (miss_analysis['compulsory_misses'] + 
                miss_analysis['capacity_misses'] + 
                miss_analysis['conflict_misses'])
    
    def _calculate_total_accesses(self, M: int, K: int, N: int) -> int:
        """Calculate total memory accesses in matrix multiplication"""
        # Each inner loop iteration does 3 accesses:
        # - Read A[i][k]
        # - Read B[k][j]
        # - Read/Write C[i][j]
        return M * K * N * 3
    
    def generate_reuse_distance_model(self, M: int, K: int, N: int, 
                                    element_size: int = 8) -> Dict:
        """
        Generate a reuse distance-based analytical model
        
        This model uses stack distance analysis for LRU caches
        """
        elements_per_block = self.b // element_size
        
        # Calculate reuse distances for each matrix
        model = {
            'reuse_analysis': {
                'matrix_A': self._analyze_A_reuse(M, K, N, elements_per_block),
                'matrix_B': self._analyze_B_reuse(M, K, N, elements_per_block),
                'matrix_C': self._analyze_C_reuse(M, K, N, elements_per_block)
            }
        }
        
        return model
    
    def _analyze_A_reuse(self, M: int, K: int, N: int, elements_per_block: int) -> Dict:
        """Analyze reuse pattern for matrix A"""
        # Each element of A is reused N times (once for each column of B)
        total_elements = M * K
        unique_blocks = (total_elements + elements_per_block - 1) // elements_per_block
        
        # Temporal reuse: each block of A is accessed N times
        # Spatial reuse: within each block, elements are accessed sequentially
        
        return {
            'total_elements': total_elements,
            'unique_blocks': unique_blocks,
            'reuse_factor': N,
            'spatial_reuse': min(elements_per_block, K),
            'temporal_reuse': N
        }
    
    def _analyze_B_reuse(self, M: int, K: int, N: int, elements_per_block: int) -> Dict:
        """Analyze reuse pattern for matrix B"""
        total_elements = K * N
        unique_blocks = (total_elements + elements_per_block - 1) // elements_per_block
        
        # Each element of B is reused M times (once for each row of A)
        return {
            'total_elements': total_elements,
            'unique_blocks': unique_blocks,
            'reuse_factor': M,
            'spatial_reuse': min(elements_per_block, N),
            'temporal_reuse': M
        }
    
    def _analyze_C_reuse(self, M: int, K: int, N: int, elements_per_block: int) -> Dict:
        """Analyze reuse pattern for matrix C"""
        total_elements = M * N
        unique_blocks = (total_elements + elements_per_block - 1) // elements_per_block
        
        # Each element of C is reused K times (accumulation)
        return {
            'total_elements': total_elements,
            'unique_blocks': unique_blocks,
            'reuse_factor': K,
            'spatial_reuse': min(elements_per_block, N),
            'temporal_reuse': K
        }
    
    def calculate_optimal_tile_sizes(self, M: int, K: int, N: int, 
                                   element_size: int = 8) -> Dict:
        """
        Calculate optimal tile sizes to minimize cache misses
        
        Uses cache capacity and associativity to determine blocking factors
        """
        # Available cache for three matrices
        cache_per_matrix = self.C // 3
        
        # Calculate maximum tile sizes
        max_A_tile = cache_per_matrix // (element_size * K)  # Tile M dimension
        max_B_tile = cache_per_matrix // (element_size * N)  # Tile K dimension
        max_C_tile = cache_per_matrix // (element_size * N)  # Tile M dimension
        
        # Ensure tiles fit in cache
        tile_M = min(M, max_A_tile, max_C_tile)
        tile_K = min(K, max_B_tile)
        tile_N = min(N, cache_per_matrix // (element_size * (tile_M + tile_K)))
        
        # Ensure tiles don't exceed associativity conflicts
        max_tile_blocks = self.assoc * self.S // 3  # Distribute across sets
        
        return {
            'optimal_tile_M': max(1, tile_M),
            'optimal_tile_K': max(1, tile_K),
            'optimal_tile_N': max(1, tile_N),
            'cache_utilization': (tile_M * K + tile_K * N + tile_M * N) * element_size / self.C
        }


def run_analysis_example():
    """Run example analysis with typical parameters"""
    
    # Example cache configuration
    cache_size = 32 * 1024  # 32KB
    block_size = 64  # 64 bytes
    associativity = 4
    
    # Calculate number of sets
    num_sets = (cache_size // block_size) // associativity
    
    # Create model
    model = CacheMissModel(cache_size, num_sets, block_size, associativity)
    
    # Example matrix sizes
    M, K, N = 512, 512, 512
    
    # Run analysis
    results = model.analyze_matrix_multiplication(M, K, N)
    reuse_model = model.generate_reuse_distance_model(M, K, N)
    tile_sizes = model.calculate_optimal_tile_sizes(M, K, N)
    
    return results, reuse_model, tile_sizes


if __name__ == "__main__":
    results, reuse_model, tile_sizes = run_analysis_example()
    
    print("=== Cache Miss Rate Analysis Model ===")
    print(f"Cache Configuration:")
    print(f"  Size: {results['cache_params']['cache_size_bytes']} bytes")
    print(f"  Sets: {results['cache_params']['num_sets']}")
    print(f"  Block Size: {results['cache_params']['block_size']} bytes")
    print(f"  Associativity: {results['cache_params']['associativity']}-way")
    
    print(f"\nMatrix Configuration:")
    print(f"  M={results['matrix_params']['M']}, K={results['matrix_params']['K']}, N={results['matrix_params']['N']}")
    print(f"  Total Data: {results['matrix_params']['total_data_bytes'] / (1024*1024):.2f} MB")
    print(f"  Working Set Ratio: {results['matrix_params']['working_set_ratio']:.2f}x cache size")
    
    print(f"\nMiss Analysis:")
    total_misses = results['miss_analysis']['total_misses']
    total_accesses = results['miss_analysis']['total_accesses']
    print(f"  Compulsory Misses: {results['miss_analysis']['compulsory_misses']:,}")
    print(f"  Capacity Misses: {results['miss_analysis']['capacity_misses']:,}")
    print(f"  Conflict Misses: {results['miss_analysis']['conflict_misses']:,}")
    print(f"  Total Misses: {total_misses:,}")
    print(f"  Total Accesses: {total_accesses:,}")
    print(f"  Overall Miss Rate: {results['miss_analysis']['overall_miss_rate']:.4f} ({results['miss_analysis']['overall_miss_rate']*100:.2f}%)")
    
    print(f"\nOptimal Tiling:")
    print(f"  Tile M: {tile_sizes['optimal_tile_M']}")
    print(f"  Tile K: {tile_sizes['optimal_tile_K']}")
    print(f"  Tile N: {tile_sizes['optimal_tile_N']}")
    print(f"  Cache Utilization: {tile_sizes['cache_utilization']:.2f}")