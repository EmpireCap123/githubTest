#!/usr/bin/env python
# coding: utf-8

# In[5]:


from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

with DAG(
    dag_id="docker_container_dag",
    start_date=days_ago(1),
    schedule_interval="0 8 * * 1-5",  # Or your desired schedule
    catchup=False,  # Important for backfills
    tags=["docker"],
) as dag:
    run_docker_container = DockerOperator(
        task_id="run_my_container",
        image="woods334/githubtest:latest",  # Replace with your image
        api_version="auto",  # Recommended
        docker_url="unix://var/run/docker.sock",  # For local Docker daemon
        # OR for other setups (Docker context, remote):
        # docker_url="tcp://<host>:<port>",
        # tls_ca_cert=</path/to/ca.pem>,
        # tls_client_cert=</path/to/cert.pem>,
        # tls_client_key=</path/to/key.pem>,
        # command="your-command-if-needed",  # Optional: Command to run inside the container
        # volumes=["/path/on/host:/path/in/container"],  # Optional: Mount volumes
        # network_mode="bridge", # Optional: Network mode
        # mem_limit="1g", # Optional: Memory limit
        # cpu_shares=512, # Optional: CPU shares
        # working_dir="/app", # Optional: Working directory inside the container
        # environment={"MY_ENV_VAR": "my_value"}, # Optional: Environment variables
        # auto_remove=True,  # Optional: Remove container after execution (good practice)
        # retries=3, # Optional: Retries on failure
        # retry_delay=timedelta(minutes=5), # Optional: Retry delay
        # log_driver="json-file", # Optional: Log driver (e.g., for persistent logging)
    )


# In[ ]:




