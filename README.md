продолжим G2
G3  проверили скрипты и данные в гугл.колабе
G4  переходим к серверным приложениям. Используем vsc
1. скачиваем данные , приготовленные в в иде оптимальной структуры
2. добавляем фичи - сохраняем
3. исследуем дрифт
исключили линии с ошибками ( убрали)   закончили на python3 G41_monitoringDrift.py
4. запускаем обучение и записываем метрики  python3 G42_11PrepareBeforeMLSave.p
5. читаем новые данные - считаем метрики на новых данных , если они поплыли
6. запускаем новое обучение - сравнивем , если лучше записываем как чемпиона  python3 G44_24CompareDataMLRet.py
7. сравниваем новые данные с преждним чемпионо



был еще 1 шаг  :
3. препроцесс   :  preprocessor
то же самое  ввод данных о сыром файле 
размещаем данные в удобном порядке 
преобразование в паркет
scr : preprocess_data_py
data_in : указываем при запуске  data/raw/data_2025-06-03        
data out :  "data/processed/processed_data.csv"

Постановка задачи с уточнениями 

1. запускаем кластер к8s  ( как в кластер поместить dockerfile-s)
2. готовим данные для докер-композе  ( на каждый шаг пишется свой Doskerfile)
3. загрузка  in compose   : downloader
готовим из сырых данных предпроцессные - очищенные , с дополнительной колонкой мид прайс= (BidPrice1+AskPrice1)/2
skript :     download_raw_data.py  
data in:     указываем data/raw/data_2025-06-03  ( данные еще для 3х дней 2025_06-04,2025_06-05,2025_06-06)
data out:    processed/processed_data_2025-06-03

4. обучение  trainer 
обогащение данных, обучение, сохранение модели и артефактов в млфлоу
scr:  train-model.py
data_in     DATA_PATH = "data/processed/processed_data_2025-06-03.csv"
data_out :  CHAMPION_DATA_PATH = "data/processed/champion_data_2025-06-03.csv"

in mlflow  : champion_data = X_train.copy()
    champion_data['mid_price'] = y_train
    os.makedirs(os.path.dirname(CHAMPION_DATA_PATH), exist_ok=True)
    champion_data.to_csv(CHAMPION_DATA_PATH, index=False)
5. monitor
scr  scripts/monitor_drift.py

.запустить все дальше - дрифт попарно  2025-06-03 ,2025_06-04,2025_06-05,2025_06-06
результаты 03-04,04-05,05-06   3 значения дрифта для среднего значения мидлпрайс

6. выводяться в млфлоу и прометеус  (в кластере)

7. запустить графану (в кластере) - получить график для 3х значений дрифта.




потом запустить автоматизация 
1. чистить файлы ,, убирать битые
2. подготовить фичи для обучения ( пока 1 фича мидл прайс )

3. если все ок опять возвращаемся к главе 5 и запускаем прометеус и графану

4. если все ок запускаем кубер и апи
5. ели ок сicd paiplain
6/ если все ок - запустить терраформ

docker pull ghcr.io/getindata/mlflow-prometheus-exporter:latest

1. запускаем докер композе
python3.9 -m venv liq-env
 source liq-env/bin/activate

остановить 
docker-compose down
docker-compose down -v

пересоздать
 docker-compose build --no-cache
 docker-compose up -d

 перезапуск 1 таски
  docker compose up -d --force-recreate --no-deps trainer
  docker compose up -d --force-recreate --no-deps mlflow
  docker compose up -d --force-recreate --no-deps postgres
  docker compose up -d --force-recreate --no-deps prometheus
  docker compose up -d --force-recreate --no-deps grafana

 
посмотреть
docker-compose logs -f mlflow
docker-compose logs -f postgres
docker-compose logs -f preprocessor
docker-compose logs trainer      # Логи обучения модели
docker-compose logs trainer --follow

docker-compose logs monitor      # Логи мониторинга дрейфа
docker-compose logs evaluator    # Логи выбора чемпиона

dipG4
подготовка данных   python3 check_data.py --date 2025-06-06


docker-compose ps

17. сделали свой экспортер :
  Dockerfile-exporter
  exporter.py

при смене exporter.py
пересобрать 
  docker-compose build mlflow-exporter
  docker-compose up -d --force-recreate mlflow-exporter
 18. полная проверка вывода данных в графану по цепочке задач :
 19. curl http://localhost:8080/metrics

 запускаем вариант для презентации
 1. экспорт данных из квика 
    - написан экспорт из квика в эксель для мониторинга 
    - написан экспорт из квика в csv вида SBER_2025_06_03 на луа для 22 акций ММВБ
 2. эагружаем данные по Сбер в гугл колаб.
    - посмотрели как они выглядят
    - провели первые тесты 
    - провели обучение на небольшом размере
    - посмотрели SHAP , какие фичи дают больший вклад в предсказание величины  проскальзывания
 3. копируем данные в яндекс клауд . Бакет создан с помощью терраформ.
 4. пишем программу для очистки данных ( убираем битые строчки) добавляем фичи 
 5. оставляем фичи мидл прайс и  ликвидность ( будем работать далее с ними)
 6. пишем проект 
    -                                          LIQUIDITY_MONITORING
    -  data venv:liq-env  mlflow  mlflow_data prometheus  trainer .env  docker-compose  README
    - data:  preprocess_2025-06-03 preprocess_2025-06-04 preprocess_2025-06-05 preprocess_2025-06-06 
    - mlflow:       Dockerfile
    - mlflow_data:  artifact mlflow.db
    - prometheus:   prometheus.yml
    - trainer:      data Dockerfile liquidity_calculator.py liquidity_results.csv requirements.txt
  7. запускаем DockerDesctop и DockerHub
  8. docker-compose build,    docker-compose up -d
  9. смотрим исполнение 
    - Mlflow        URL: http://localhost:5000
    - Pushgateway:  URL: http://localhost:9091
    - Prometheus:   URL: http://localhost:9090
    - Grafana:      URL: http://localhost:3000
  10. строим Дашборды в Графане  







#   L i q u i d i t y _ O t u s D i p l o m 2 0 2 5 
 
 
