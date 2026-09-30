"""Flask API for the forecast, telemetry, and scheduler webhook."""

from datetime import datetime
from flask import Flask, jsonify, render_template, request

from module1 import get_current_carbon_intensity, record_live_observation, run_forecast
from module2_telemetry_sim import SimulatedEdgeNode
from module3_orchestrator import SpatioTemporalThermalOrchestrator
from module4_quantization_sim import WorkloadTask

app = Flask(__name__)
orchestrator = SpatioTemporalThermalOrchestrator()
nodes = {f'pi-node-{index}': SimulatedEdgeNode(f'pi-node-{index}') for index in range(1, 4)}


def _node_from_payload(payload):
    name = payload.get('node_id', payload.get('name'))
    node = nodes.setdefault(name, SimulatedEdgeNode(name))
    if 'temperature_celsius' in payload:
        node.temp = float(payload['temperature_celsius'])
    if 'power_watts' in payload:
        node.current_power = float(payload['power_watts'])
    node.is_busy = bool(payload.get('is_busy', node.is_busy))
    return node


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/forecast')
def api_forecast():
    try:
        result = run_forecast()
        result.update(status='ok', updated_at=datetime.now().isoformat(timespec='seconds'))
        return jsonify(result)
    except Exception as error:
        return jsonify(status='error', message=str(error)), 500


@app.route('/api/observation', methods=['POST'])
def api_observation():
    payload = request.get_json(silent=True) or {}
    if 'timestamp' not in payload or 'value' not in payload:
        return jsonify(status='error', message='timestamp and value are required'), 400
    try:
        record_live_observation(payload['timestamp'], payload['value'])
        return jsonify(status='ok', message='Observation added to the training dataset')
    except (TypeError, ValueError, KeyError) as error:
        return jsonify(status='error', message=str(error)), 400


@app.route('/api/telemetry', methods=['GET', 'POST'])
def api_telemetry():
    if request.method == 'POST':
        payload = request.get_json(silent=True) or {}
        if not payload.get('node_id') and not payload.get('name'):
            return jsonify(status='error', message='node_id is required'), 400
        _node_from_payload(payload)
    return jsonify([node.get_telemetry() for node in nodes.values()])


@app.route('/metrics')
def metrics():
    """Expose node data in Prometheus text format for scraping."""
    lines = [
        '# HELP edge_node_temperature_celsius Current simulated node temperature.',
        '# TYPE edge_node_temperature_celsius gauge',
        '# HELP edge_node_power_watts Current simulated node power.',
        '# TYPE edge_node_power_watts gauge',
        '# HELP edge_node_busy Whether a node is running a workload.',
        '# TYPE edge_node_busy gauge',
    ]
    for node in nodes.values():
        telemetry = node.get_telemetry()
        label = f'node="{node.node_id}"'
        lines.extend([
            f'edge_node_temperature_celsius{{{label}}} {telemetry["temperature_celsius"]}',
            f'edge_node_power_watts{{{label}}} {telemetry["power_watts"]}',
            f'edge_node_busy{{{label}}} {int(telemetry["is_busy"])}',
        ])
    return '\n'.join(lines) + '\n', 200, {'Content-Type': 'text/plain; version=0.0.4'}


def _schedule_payload(payload):
    task_data = payload.get('task', {})
    task = WorkloadTask(task_data.get('task_id', 'webhook-task'),
                        task_data.get('is_latency_critical', True),
                        task_data.get('accuracy_floor', 0.85))
    carbon = get_current_carbon_intensity(payload.get('current_grid_carbon'))
    selected, status = orchestrator.schedule(task, list(nodes.values()), carbon)
    return selected, status, task


@app.route('/api/scheduler/filter', methods=['POST'])
def scheduler_filter():
    payload = request.get_json(silent=True) or {}
    requested = {item.get('metadata', {}).get('name') for item in payload.get('nodes', [])}
    allowed = [node.node_id for node in nodes.values()
               if node.node_id in requested and node.temp < orchestrator.temp_limit]
    return jsonify({'nodes': [{'metadata': {'name': name}} for name in allowed]})


@app.route('/api/scheduler/prioritize', methods=['POST'])
def scheduler_prioritize():
    payload = request.get_json(silent=True) or {}
    carbon = float(payload.get('current_grid_carbon', 0))
    requested = [item.get('metadata', {}).get('name') for item in payload.get('nodes', [])]
    ranked = orchestrator.score_nodes([nodes[name] for name in requested if name in nodes], carbon)
    scores = {node.node_id: max(0, round((1 - penalty) * 10)) for node, penalty in ranked}
    return jsonify([{'host': name, 'score': scores.get(name, 0)} for name in requested])


@app.route('/api/scheduler/schedule', methods=['POST'])
def scheduler_schedule():
    selected, status, task = _schedule_payload(request.get_json(silent=True) or {})
    return jsonify({'node': selected.node_id if selected else None,
                    'status': status, 'precision': task.precision})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)