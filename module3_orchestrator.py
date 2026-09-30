class SpatioTemporalThermalOrchestrator:
    def __init__(self, w1=0.45, w2=0.35, w3=0.20, temp_limit=75.0):
        # Configurable weights balancing carbon, thermal state, and power
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.temp_limit = temp_limit

    def filter_nodes(self, nodes):
        """Filter Phase: Eliminate nodes exceeding thermal threshold."""
        viable = [node for node in nodes if node.temp < self.temp_limit]
        return viable

    def score_nodes(self, viable_nodes, current_grid_carbon, max_observed_carbon=800.0):
        """
        Score Phase: Lower score indicates lower environmental and hardware impact.
        Normalized cost penalty = w1*C_norm + w2*T_norm + w3*P_norm
        """
        scored_nodes = []
        c_norm = min(1.0, current_grid_carbon / max_observed_carbon)

        for node in viable_nodes:
            t_norm = (node.temp - node.ambient_temp) / (self.temp_limit - node.ambient_temp)
            p_norm = node.current_power / 12.0  # Normalized to max Pi power
            
            penalty = (self.w1 * c_norm) + (self.w2 * t_norm) + (self.w3 * p_norm)
            scored_nodes.append((node, penalty))

        # Sort ascending (lowest penalty score wins)
        scored_nodes.sort(key=lambda x: x[1])
        return scored_nodes

    def schedule(self, task, nodes, current_grid_carbon):
        viable = self.filter_nodes(nodes)
        
        # All nodes hot or saturated
        if not viable:
            if task.is_latency_critical:
                if task.scale_precision_to_int8():
                    # Fallback retry with lowest temp node available
                    cooler_node = min(nodes, key=lambda n: n.temp)
                    return cooler_node, "SCHEDULED_QUANTIZED_INT8"
                else:
                    return None, "REJECTED_ACCURACY_FLOOR_BREACH"
            else:
                return None, "DEFERRED_WAITING_COOL_CLEAN_WINDOW"

        ranked = self.score_nodes(viable, current_grid_carbon)
        selected_node = ranked[0][0]
        return selected_node, "SCHEDULED_OPTIMAL_FP32"