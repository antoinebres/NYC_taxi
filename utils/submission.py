import datetime
import re
import pandas as pd
from utils.config import *


def write(df):
    base = "NYtaxi_"
    timestamp = str(datetime.datetime.now())[:-7]
    timestamp = re.sub('-', '_', timestamp)
    timestamp = re.sub(':', '_', timestamp)
    timestamp = re.sub(' ', '_', timestamp)
    filename = base + timestamp
    filelocation = "/".join([SUBMISSIONS_DIR, filename])
    df.to_csv(filelocation)
