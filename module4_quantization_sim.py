"""Mock dynamic precision scaling used by the simulation."""

class WorkloadTask:
    def __init__(self, task_id, is_latency_critical=True, accuracy_floor=0.85, base_duration_seconds=10):
        self.task_id = task_id
        self.is_latency_critical = is_latency_critical
        self.accuracy_floor = accuracy_floor
        self.base_duration_seconds = base_duration_seconds
        self.precision = 'FP32'
        self.expected_accuracy = 0.94

    def scale_precision_to_int8(self):
        int8_accuracy = 0.90
        if int8_accuracy < self.accuracy_floor:
            return False
        self.precision = 'INT8'
        self.expected_accuracy = int8_accuracy
        return True