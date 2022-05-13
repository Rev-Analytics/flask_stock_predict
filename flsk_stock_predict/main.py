from flask import (
    Flask,
    render_template, 
    redirect,
    request, 
    url_for, 
    send_file)
from flask_wtf import FlaskForm
import os
from os import listdir
from os.path import isfile, join

from google.cloud import storage
from create_config import bucketName


import time
from wtforms import StringField, IntegerField,  DateTimeField
#SubmitField, DateField, SelectField,   SelectMultipleField,
from wtforms.validators import DataRequired
import pandas as pd

# import tensorflow as tf
# from tensorflow import keras
# import tensorflow_datasets as tfds
# from tensorflow.keras import models, layers
# from tensorflow.keras.layers import Conv1D,BatchNormalization, Dropout, Flatten, Input, Dense
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.keras.models import Sequential

from get_sp_data import stock_data_pull, fill_blanks, fill_blanks, pre_proc_df, create_input_df

app = Flask(__name__)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

# input window and forecast horizon
window = 70
horizon = 10
theta = window+horizon

history = '6mo'

@app.route("/",methods=['GET','POST'])
def main():
    if request.method == 'POST':
        redirect('index.html')
    else:
        return render_template('main.html')

@app.route("/index", methods=['GET','POST'] )
def index():
    stock_total = stock_data_pull(history)
    stock_ready = pre_proc_df(stock_total)

    # checks to ensure if all tckrs have same size
    sizes = stock_ready.groupby('Name').size().unique()
    size_count = sizes.shape[0]

    # Checks that all stocks have the same number of observations
    sizes = stock_ready.groupby('Name').size().unique()
    size_count = sizes.shape[0]
    if size_count == 1:
        print(f'all good, all data contain {sizes[0]} observations')
    else:
        print(f'error, your stock contain tckrs with varying number of observations:{sizes}')

    stock_pred = create_input_df(stock_ready,window)

    #print(f'this is the dataframe used as input for model {stock_pred.head()}')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # save the file into the static folder
    stock_pred.to_csv(f'stock_pred_{timestr}.csv')
    print('save input ready file')
        # file.save(filepath)
        # # try and save only json
        # #file.save(filepath_json)

        # new_filename = file.filename.replace('.csv','.json')
        # json_name = 'static/'+new_filename
        # filepath_json = os.path.join(json_name)
       
        # # read in the same exact file
        # df_csv = pd.read_csv(csv_name)

        # df_csv.to_json(filepath_json)
        # #print('bucketfolder',bucketFolder)

        # os.remove(csv_name)

        # blob = bucket.blob(new_filename)

        # blob.upload_from_filename(filepath_json)

        # # check that local file made it to the bucket
        # files = bucket.list_blobs()
        # print('files',files)
        # fileList = [file.name for file in files if '.' in file.name]
        # # delete local file after check
        # print('bucket filelist',fileList)
        # if file.filename in fileList:
        #     os.remove(filepath)   
        #     print(f'{file.filename} deleted from bucket.')

        # return redirect(url_for('data'))
    return render_template('index.html')

@app.route("/data", methods =['POST','GET'] )
def data():
    return render_template('sent.html')


# if __name__ == '__main__':
#     print('got to the end')
#     app.run(debug = True)

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8081)))    

if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 8081)))    
