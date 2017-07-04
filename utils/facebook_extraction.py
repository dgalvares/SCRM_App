import facebook
import json
import pprint
from pymongo import MongoClient
import gridfs
from time import sleep

client = MongoClient()
db = client.socialNet
fs = gridfs.GridFS(db)
access_token_Coleta_API = "1493379330957167|mr9g7FMiztrz0dPlp1xbSjvotms"

access_token_Colea = "1633758386640988|jUPFAXut2ObAiEGP_Dqll7mSb9s"
# app_id = "1493379330957167"
# app_secret = "36c77965c64bc3301800202d148e4fc8"

graph = facebook.GraphAPI(access_token_Colea)
arguments = {}
page = "netflixbrasil"

requesttxt = "https://graph.facebook.com/v2.6/%s/posts?summary=1&filter=stream&fields=likes.summary(true),comments.summary(true),reactions.type(LOVE).summary(true),shares,message,from,type,picture,description,created_time"

def collect(self):
    count = 1
    request = self.get_connections(page,'posts?summary=1&filter=stream&fields=likes.summary(true),comments.summary(true),reactions.summary(true),shares,message,from,type,status_type,picture,link,source,name,caption,description,icon,created_time',**arguments)
    for item in request['data']:
        pprint.pprint((item))

        print('>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<')
        print(count)
        face = db.facebook.insert_one(item)
        print(face.inserted_id)
        count = count +1
collect(graph)