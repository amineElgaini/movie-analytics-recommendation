from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow"

default_args = {
    "owner": "amine",
    "retries": 1,
}

with DAG(
    dag_id="movie_intelligence_pipeline",
    default_args=default_args,
    start_date=datetime(2026, 9, 28),
    schedule=None,
    catchup=False,
) as dag:

    extract = BashOperator(
        task_id="extract",
        bash_command=f"cd {PROJECT_DIR} && python src/extract.py",
    )
    clean = BashOperator(
        task_id="clean",
        bash_command=f"cd {PROJECT_DIR} && python src/clean.py",
    )
    features = BashOperator(
        task_id="features",
        bash_command=f"cd {PROJECT_DIR} && python src/features.py",
    )
    mongo_store = BashOperator(
        task_id="mongo_store",
        bash_command=f"cd {PROJECT_DIR} && python src/mongo_store.py",
    )
    train = BashOperator(
        task_id="train",
        bash_command=f"cd {PROJECT_DIR} && python src/train.py",
    )

    extract >> clean >> features >> mongo_store >> train