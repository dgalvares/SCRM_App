import facebook
import json
import pprint
from pymongo import MongoClient
from utils.db_connection import MongodbConnector
import gridfs
from time import sleep

db = MongodbConnector.connect_db('socialNet')
collection = MongodbConnector.connect_collection(db, 'facebook')
fs = gridfs.GridFS(db)
access_token_Coleta_API = "1493379330957167|mr9g7FMiztrz0dPlp1xbSjvotms"

access_token_Coleta = "1633758386640988|jUPFAXut2ObAiEGP_Dqll7mSb9s"
# app_id = "1493379330957167"
# app_secret = "36c77965c64bc3301800202d148e4fc8"

graph = facebook.GraphAPI(access_token_Coleta)
arguments = {}
page = "netflixbrasil"

# requesttxt = "https://graph.facebook.com/v2.6/%s/posts?summary=1&filter=stream&fields=likes.summary(true),comments.summary(true),reactions.type(LOVE).summary(true),shares,message,from,type,picture,description,created_time"


def collect(graph,filter, extracao_id, extracao):
    count = 1
    request = graph.get_connections(filter,'posts?summary=1&filter=stream&fields=likes.summary(true),comments.summary(true),reactions.summary(true),shares,message,from,type,status_type,picture,link,source,name,caption,description,icon,created_time',**arguments)
    for item in request['data']:
        pprint.pprint(item['comments'])
        face = json.loads(item)
        face['extracao']=extracao_id
        print(">>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<")
        collection.insert_one(item)
        print(face.inserted_id)
        count = count +1
# collect(graph)


def extrair(filter, extracao_id, extracao):
    collect(graph, filter, extracao_id, extracao)

extrair()