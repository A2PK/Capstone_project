import csv
import time
import random
import statistics
import json
from datetime import datetime, timezone
from confluent_kafka import Producer
import os

# Kafka config (from .env, hardcoded here for demo; in prod, use os.getenv or dotenv)
KAFKA_BROKER = 'pkc-ldvr1.asia-southeast1.gcp.confluent.cloud:9092'
KAFKA_TOPIC = 'water-quality-realtime'
KAFKA_API_KEY = '5KGFM2IBM3UZ6OJL'
KAFKA_API_SECRET = 'c4eX8sutPRpN8k8pnexsYS+Ryy65KPF5Xf6nCckokUmf31oTrBlbSHV6m/c8bjCP'

# Column names in seed.csv
PH_COL = 'pH'
DO_COL = 'DO'
EC_COL = 'Độ dẫn'
STATION_NAME_COL = 'Điểm Quan Trắc'

# Helper to parse float with comma or dot
def parse_float(val):
    if isinstance(val, str):
        val = val.replace(',', '.')
    try:
        return float(val)
    except Exception:
        return None

def extract_column_indices(header):
    ph_idx = header.index(PH_COL)
    do_idx = header.index(DO_COL)
    ec_idx = header.index(EC_COL)
    station_idx = header.index(STATION_NAME_COL)
    return ph_idx, do_idx, ec_idx, station_idx

# Read per-station means and stddevs from seed.csv
def get_station_stats_from_csv(csv_path):
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        if not any(h == PH_COL for h in header):
            header = next(reader)
        ph_idx, do_idx, ec_idx, station_idx = extract_column_indices(header)
        station_data = {}
        for row in reader:
            if len(row) <= max(ph_idx, do_idx, ec_idx, station_idx):
                continue
            station = row[station_idx].strip()
            ph = parse_float(row[ph_idx])
            do = parse_float(row[do_idx])
            ec = parse_float(row[ec_idx])
            if not station:
                continue
            if station not in station_data:
                station_data[station] = {'pH': [], 'DO': [], 'EC': []}
            if ph is not None:
                station_data[station]['pH'].append(ph)
            if do is not None:
                station_data[station]['DO'].append(do)
            if ec is not None:
                station_data[station]['EC'].append(ec)
        # Compute stats
        station_stats = {}
        for station, vals in station_data.items():
            try:
                station_stats[station] = {
                    'pH': (statistics.mean(vals['pH']), statistics.stdev(vals['pH']) if len(vals['pH']) > 1 else 0.1),
                    'DO': (statistics.mean(vals['DO']), statistics.stdev(vals['DO']) if len(vals['DO']) > 1 else 0.1),
                    'EC': (statistics.mean(vals['EC']), statistics.stdev(vals['EC']) if len(vals['EC']) > 1 else 0.1),
                }
            except Exception:
                continue
        return station_stats

def generate_random_value(mean, stdev, min_val=None, max_val=None):
    val = random.gauss(mean, stdev)
    if min_val is not None:
        val = max(val, min_val)
    if max_val is not None:
        val = min(val, max_val)
    return round(val, 2)

def load_stations(stations_csv):
    stations = []
    with open(stations_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stations.append({
                'id': row['id'],
                'name': row['name'],
                'location': row['location'],
            })
    return stations

def make_kafka_producer():
    conf = {
        'bootstrap.servers': KAFKA_BROKER,
        'security.protocol': 'SASL_SSL',
        'sasl.mechanisms': 'PLAIN',
        'sasl.username': KAFKA_API_KEY,
        'sasl.password': KAFKA_API_SECRET,
        'client.id': 'python-producer',
    }
    return Producer(conf)

def build_message(station, ph, do, ec):
    now = datetime.now(timezone.utc).isoformat()
    features = [
        {'name': 'pH', 'value': ph},
        {'name': 'DO', 'value': do},
        {'name': 'EC', 'value': ec},
    ]
    return {
        'monitoring_time': now,
        'station_id': station['id'],
        'station_name': station['name'],
        'source': 'demo-producer',
        'observation_type': 'realtime-monitoring',
        'features': features,
    }

def main():
    station_stats = get_station_stats_from_csv('seed.csv')
    stations = load_stations('stations.csv')
    producer = make_kafka_producer()
    print(f'Loaded {len(stations)} stations, {len(station_stats)} with stats')
    while True:
        for station in stations:
            name = station['name']
            stats = station_stats.get(name)
            if not stats:
                continue
            ph = generate_random_value(*stats['pH'], min_val=0, max_val=14)
            do = generate_random_value(*stats['DO'], min_val=0)
            ec = generate_random_value(*stats['EC'], min_val=0)
            msg = build_message(station, ph, do, ec)
            producer.produce(KAFKA_TOPIC, json.dumps(msg).encode('utf-8'))
            print('Produced:', msg)
        producer.flush()
        time.sleep(10)

if __name__ == '__main__':
    main()
