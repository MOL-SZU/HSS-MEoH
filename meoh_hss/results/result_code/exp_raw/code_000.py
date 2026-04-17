import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    
    def greedy_select(candidate_points, select_k, ref_point):
        if len(candidate_points) <= select_k:
            return candidate_points
        
        selected = np.empty((0, points.shape[1]))
        remaining = candidate_points.copy()
        
        while len(selected) < select_k:
            if len(selected) == 0:
                hv_contrib = np.prod(ref_point - remaining, axis=1)
                best_idx = np.argmax(hv_contrib)
            else:
                hv_base = np.prod(ref_point - np.max(selected, axis=0))
                hv_contribs = []
                for cand in remaining:
                    temp_set = np.vstack([selected, cand])
                    hv_new = np.prod(ref_point - np.max(temp_set, axis=0))
                    hv_contribs.append(hv_new - hv_base)
                best_idx = np.argmax(hv_contribs)
            
            selected = np.vstack([selected, remaining[best_idx]])
            remaining = np.delete(remaining, best_idx, axis=0)
        
        return selected
    
    def hierarchical_clustering(pts, max_clusters):
        if len(pts) <= max_clusters:
            return [pts]
        
        clusters = [pts.copy()]
        
        while len(clusters) < max_clusters:
            largest_idx = np.argmax([len(c) for c in clusters])
            largest_cluster = clusters[largest_idx]
            
            if len(largest_cluster) <= 1:
                break
            
            centroid = np.mean(largest_cluster, axis=0)
            distances = np.linalg.norm(largest_cluster - centroid, axis=1)
            median_dist = np.median(distances)
            
            mask1 = distances <= median_dist
            mask2 = distances > median_dist
            
            subcluster1 = largest_cluster[mask1]
            subcluster2 = largest_cluster[mask2]
            
            del clusters[largest_idx]
            if len(subcluster1) > 0:
                clusters.append(subcluster1)
            if len(subcluster2) > 0:
                clusters.append(subcluster2)
        
        return clusters
    
    if k >= len(points):
        return points
    
    max_clusters = min(k, int(np.sqrt(len(points))))
    clusters = hierarchical_clustering(points, max_clusters)
    
    cluster_allocations = np.zeros(len(clusters), dtype=int)
    cluster_sizes = np.array([len(c) for c in clusters])
    
    for i in range(len(clusters)):
        cluster_allocations[i] = max(1, int(k * cluster_sizes[i] / len(points)))
    
    remaining_k = k - np.sum(cluster_allocations)
    if remaining_k > 0:
        while remaining_k > 0:
            density_ratios = cluster_sizes / (cluster_allocations + 1)
            max_density_idx = np.argmax(density_ratios)
            cluster_allocations[max_density_idx] += 1
            remaining_k -= 1
    
    selected_subsets = []
    for i, cluster in enumerate(clusters):
        local_ref = np.minimum(reference_point, np.max(cluster, axis=0) * 1.1)
        subset = greedy_select(cluster, cluster_allocations[i], local_ref)
        selected_subsets.append(subset)
    
    final_selected = np.vstack(selected_subsets)
    
    if len(final_selected) > k:
        final_selected = greedy_select(final_selected, k, reference_point)
    elif len(final_selected) < k:
        remaining_points = np.array([p for p in points if not any(np.array_equal(p, sp) for sp in final_selected)])
        additional = greedy_select(remaining_points, k - len(final_selected), reference_point)
        final_selected = np.vstack([final_selected, additional])
    
    return final_selected

