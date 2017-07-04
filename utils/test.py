from utils import twitter_stream_extraction
import threading
import time

tempo = time.time().__str__()
tse = twitter_stream_extraction
threading._start_new_thread(tse.extrair,("Londres",10,110,time))