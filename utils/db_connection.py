from pymongo import MongoClient
from urllib.parse import quote

class MongodbConnector:
    def connect_db(db_name):
        senha = quote('scrm150')
        db_client = MongoClient('mongodb://scrm:' + senha + '@localhost:27017')

        db = db_client[db_name]
        return db

    def connect_collection(db,collection):
        db_collection = db[collection]
        return db_collection
