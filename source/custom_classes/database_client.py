import os
import pathlib
import pandas as pd

from dotenv import load_dotenv
from pymongo import MongoClient


class DatabaseClient:
    def __init__(self, secrets_path: str = pathlib.Path(__file__).parent.joinpath('..', '..', 'configs', 'secrets.env')):
        load_dotenv(secrets_path)  # Take environment variables from .env

        # Provide the mongodb atlas url to connect python to mongodb using pymongo
        self.connection_string = os.getenv("CONNECTION_STRING")
        self.client = None
        self.collection = None

    def connect(self, collection_name):
        # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
        self.client = MongoClient(self.connection_string)
        self.collection = self.client[os.getenv("DB_NAME")][collection_name]

    def execute_write_query(self, records):
        return self.collection.insert_many(records)

    def execute_read_query(self, query: dict):
        cursor = self.collection.find(query)
        records = []
        for record in cursor:
            del record['_id']
            records.append(record)

        return records

    def read_model_metric_dfs_from_db(self, session_uuid):
        records = self.execute_read_query(query={'session_uuid': session_uuid, 'tag': 'OK'})
        model_metric_dfs = pd.DataFrame(records)

        # Capitalize column names to be consistent across the whole library
        new_column_names = []
        for col in model_metric_dfs.columns:
            new_col_name = '_'.join([c.capitalize() for c in col.split('_')])
            new_column_names.append(new_col_name)

        model_metric_dfs.columns = new_column_names
        return model_metric_dfs

    def get_db_writer(self):
        def db_writer_func(run_models_metrics_df, collection=self.collection):
            # Rename Pandas columns to lower case
            run_models_metrics_df.columns = run_models_metrics_df.columns.str.lower()
            collection.insert_many(run_models_metrics_df.to_dict('records'))

        return db_writer_func

    def close(self):
        self.client.close()
