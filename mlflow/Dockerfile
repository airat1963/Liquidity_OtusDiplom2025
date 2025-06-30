FROM python:3.9
RUN pip install mlflow==2.9.2
EXPOSE 5000
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:////home/mlflowuser/mlflow.db","--default-artifact-root", "/home/mlflowuser/artifacts","--host", "0.0.0.0"]