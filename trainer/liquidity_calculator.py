import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import mlflow
from dotenv import load_dotenv

load_dotenv()

# Конфигурация
PUSHGATEWAY = os.getenv('PROMETHEUS_PUSHGATEWAY_URL')
MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI')
DATA_PATH = os.getenv('DATA_PATH', '/app/data/preprocess')
EMA_SPAN = 12  # Для экспоненциального среднего (примерно 2 минуты)

def calculate_liquidity(row):
    """Расчет ликвидности для одной строки стакана"""
    bid_liquidity = sum(row[f'BidQty{i}'] for i in range(1, 11))
    ask_liquidity = sum(row[f'AskQty{i}'] for i in range(1, 11))
    return bid_liquidity + ask_liquidity

def process_day(file_path):
    """Обработка данных за один день с учетом новых требований"""
    try:
        # Извлекаем дату из имени файла
        date_str = os.path.basename(file_path).split('_')[1].split('.')[0]
        
        # Загрузка данных
        df = pd.read_csv(file_path)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.sort_values('DateTime')
        
        # Пропускаем первые 6 строк дня
        df = df.iloc[6:]
        
        # Рассчитываем ликвидность для каждой строки
        df['liquidity'] = df.apply(calculate_liquidity, axis=1)
        
        # Создаем колонку для минутных агрегатов
        df['minute_group'] = (df.index // 6).astype(int)
        
        # Вычисляем среднюю ликвидность каждые 6 строк (1 минута)
        minute_avg = df.groupby('minute_group')['liquidity'].mean().reset_index()
        minute_avg.rename(columns={'liquidity': 'minute_avg'}, inplace=True)
        
        # Соединяем с основным DataFrame
        df = pd.merge(df, minute_avg, on='minute_group', how='left')
        
        # Заполняем пропуски предыдущим значением
        df['minute_avg'] = df['minute_avg'].ffill()
        
        # Применяем экспоненциальное скользящее среднее
        df['ema_liquidity'] = df['minute_avg'].ewm(span=EMA_SPAN, adjust=False).mean()
        
        # Рассчитываем метрики дня
        day_mean = df['ema_liquidity'].mean()
        day_variance = df['ema_liquidity'].var()
        
        return date_str, {
            'mean_liquidity': day_mean,
            'variance_liquidity': day_variance,
            'mid_price': df['mid_price'].iloc[-1]
        }
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None, None

if __name__ == '__main__':
    # Инициализация MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Liquidity Monitoring")
    
    # Список файлов для обработки
    files = [
        f"{DATA_PATH}/preprocess_2025-06-03.csv",
        f"{DATA_PATH}/preprocess_2025-06-04.csv",
        f"{DATA_PATH}/preprocess_2025-06-05.csv",
        f"{DATA_PATH}/preprocess_2025-06-06.csv"
    ]
    
    registry = CollectorRegistry()
    results = {}
    mid_prices = {}
    
    with mlflow.start_run():
        # Обработка каждого файла
        for file in files:
            if os.path.exists(file):
                date, metrics = process_day(file)
                if date and metrics:
                    results[date] = metrics
                    mid_prices[date] = metrics['mid_price']
                    
                    # Логирование в MLflow
                    mlflow.log_metrics({
                        f"{date}_mean": metrics['mean_liquidity'],
                        f"{date}_variance": metrics['variance_liquidity'],
                        f"{date}_mid_price": metrics['mid_price']
                    })
                    print(f"Processed {date}: Mean={metrics['mean_liquidity']:.2f}, Var={metrics['variance_liquidity']:.2f}")
            else:
                print(f"File not found: {file}")
        
        # Регистрация метрик в Prometheus
        mean_gauge = Gauge('orderbook_mean_liquidity', 'Mean liquidity by date', ['date'], registry=registry)
        var_gauge = Gauge('orderbook_variance_liquidity', 'Liquidity variance by date', ['date'], registry=registry)
        mid_price_gauge = Gauge('orderbook_mid_price', 'Mid price by date', ['date'], registry=registry)
        
        for date, values in results.items():
            mean_gauge.labels(date=date).set(values['mean_liquidity'])
            var_gauge.labels(date=date).set(values['variance_liquidity'])
            mid_price_gauge.labels(date=date).set(values['mid_price'])
        
        # Расчет средних значений по дням
        avg_mean = np.mean([v['mean_liquidity'] for v in results.values()])
        avg_variance = np.mean([v['variance_liquidity'] for v in results.values()])
        avg_mid = np.mean(list(mid_prices.values()))
        
        # Регистрация средних значений
        Gauge('orderbook_avg_mean', '4-day average mean liquidity', registry=registry).set(avg_mean)
        Gauge('orderbook_avg_variance', '4-day average liquidity variance', registry=registry).set(avg_variance)
        Gauge('orderbook_avg_mid_price', '4-day average mid price', registry=registry).set(avg_mid)
        
        # Логирование средних значений в MLflow
        mlflow.log_metrics({
            "avg_mean": avg_mean,
            "avg_variance": avg_variance,
            "avg_mid_price": avg_mid
        })
        
        # Отправка метрик в Pushgateway
        push_to_gateway(PUSHGATEWAY, job='liquidity_calculation', registry=registry)
        print(f"Metrics pushed to Pushgateway at {PUSHGATEWAY}")
        
        # Сохранение результатов в MLflow
        results_df = pd.DataFrame(results).T
        results_df.to_csv("liquidity_results.csv")
        mlflow.log_artifact("liquidity_results.csv")
        
        print("Processing completed successfully!")
        time.sleep(10)