"""Deterministic LSTM carbon-intensity forecast service."""

from datetime import datetime, timedelta
import json
import os
from urllib.request import Request, urlopen
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

DATA_PATH = Path(__file__).with_name('india_monthly_full_release_long_format.csv')
PREDICTION_HISTORY_PATH = Path(__file__).with_name('india_forecast_predictions.csv')
_FORECAST_CACHE = {}


def get_current_carbon_intensity(fallback=None):
    """Read Electricity Maps when configured, otherwise use the forecast."""
    token = os.environ.get('ELECTRICITY_MAPS_TOKEN')
    zone = os.environ.get('ELECTRICITY_MAPS_ZONE', 'IN')
    if token:
        request = Request(
            f'https://api.electricitymaps.com/v3/carbon-intensity/latest?zone={zone}',
            headers={'auth-token': token},
        )
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode('utf-8'))
        return float(payload['carbonIntensity'])
    if fallback is not None:
        return float(fallback)
    return float(run_forecast()['forecast'][0])


class CarbonForecasterLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=24):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, values):
        hidden = torch.zeros(self.num_layers, values.size(0), self.hidden_size, device=values.device)
        cell = torch.zeros_like(hidden)
        output, _ = self.lstm(values, (hidden, cell))
        return self.fc(output[:, -1, :])


def record_live_observation(timestamp, value):
    timestamp = pd.to_datetime(timestamp, errors='raise')
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError('value must be a finite non-negative number')
    dataset = pd.read_csv(DATA_PATH)
    dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')
    row = {column: '' for column in dataset.columns}
    row.update({'Date': timestamp, 'Country': 'India', 'State': 'India',
                'Variable': 'Total emissions', 'Value': value})
    duplicate = ((dataset['Date'] == timestamp) & (dataset['Country'] == 'India') &
                 (dataset['State'] == 'India') & (dataset['Variable'] == 'Total emissions'))
    dataset = pd.concat([dataset.loc[~duplicate], pd.DataFrame([row])], ignore_index=True)
    dataset.sort_values('Date').to_csv(DATA_PATH, index=False, date_format='%Y-%m-%dT%H:%M:%S')
    _FORECAST_CACHE.clear()


def _record_prediction_history(result):
    generated_at = datetime.now().replace(second=0, microsecond=0)
    rows = pd.DataFrame({
        'generated_at': [generated_at] * len(result['forecast']),
        'forecast_for': [generated_at + timedelta(hours=i) for i in range(1, 25)],
        'Country': result['state'], 'Variable': result['variable'],
        'Value': result['forecast'], 'Value_type': 'prediction',
    })
    rows.to_csv(PREDICTION_HISTORY_PATH, mode='a',
                header=not PREDICTION_HISTORY_PATH.exists(), index=False)


def run_forecast():
    data_version = DATA_PATH.stat().st_mtime_ns
    if data_version in _FORECAST_CACHE:
        result = _FORECAST_CACHE[data_version].copy()
        result['forecast_source'] = 'cache'
        return result

    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    data = pd.read_csv(DATA_PATH)
    filtered = data[(data['Country'] == 'India') & (data['Variable'] == 'Total emissions')].copy()
    if filtered.empty:
        raise ValueError('The dataset contains no India Total emissions records.')
    date_column = next((name for name in ('Date', 'Month', 'Year') if name in filtered), 'Date')
    filtered[date_column] = pd.to_datetime(filtered[date_column])
    filtered['Value'] = pd.to_numeric(filtered['Value'], errors='coerce')
    series = (filtered.sort_values(date_column).drop_duplicates(date_column)
              .set_index(date_column)['Value'].dropna().resample('h').interpolate().dropna())
    # Keep the simulation responsive while retaining more than enough context
    # for 24-step input and output windows.
    series = series.tail(2000)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(series.to_numpy().reshape(-1, 1))
    sequence_length = 24
    prediction_length = 24
    features, targets = [], []
    for index in range(len(scaled) - sequence_length - prediction_length + 1):
        features.append(scaled[index:index + sequence_length])
        targets.append(scaled[index + sequence_length:index + sequence_length + prediction_length])
    if len(features) < 10:
        raise ValueError('Not enough historical observations to train the forecast model.')
    x_train = torch.tensor(np.asarray(features), dtype=torch.float32)
    y_train = torch.tensor(np.asarray(targets), dtype=torch.float32).squeeze(-1)
    model = CarbonForecasterLSTM(output_size=prediction_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.MSELoss()
    for _ in range(20):
        optimizer.zero_grad()
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        recent = torch.tensor(scaled[-sequence_length:], dtype=torch.float32).unsqueeze(0)
        prediction = torch.clamp(model(recent), -1, 1).numpy()
    forecast = np.round(scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(-1), 2)
    result = {'state': 'India', 'variable': 'Total emissions',
              'forecast': [float(value) for value in forecast],
              'labels': [f'Hour {i}' for i in range(1, 25)],
              'forecast_source': 'retrained'}
    _FORECAST_CACHE[data_version] = result
    _record_prediction_history(result)
    return result.copy()