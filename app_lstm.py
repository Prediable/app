from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
import keras
import psycopg2
import psycopg2.extras
from sklearn.preprocessing import MinMaxScaler

hostname = 'uwe-azure.postgres.database.azure.com'
database = 'lstm_data'
username = 'Jan_17026846'
password = 'Jasiu995'
port_id = 5432

conn = None
cur = None

fetched_data = []

app = Flask(__name__)
model = keras.models.load_model("lstm.h5")


@app.route('/predict', methods=['GET'])
def predict():
    a = get_prediction()
    return jsonify(a)


def connect_to_db():
    try:
        conn = psycopg2.connect(
            host=hostname,
            database=database,
            user=username,
            password=password,
            port=port_id)
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        fetch_data = 'SELECT * FROM PUBLIC."Feed"'
        cur.execute(fetch_data)
        for record in cur.fetchall():
            fetched_data.append(record)

    except Exception as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
        cur.close()
        conn.close()


def get_prediction():
    connect_to_db()
    model = keras.models.load_model("lstm.h5")
    df = pd.DataFrame(fetched_data)
    df.columns = ['Month', 'Month.1', 'Total Crimes']
    df.index = pd.to_datetime(df['Month'], format='%Y.%m')
    df = df.drop('Month', 1)
    df = df.drop('Month.1', 1)
    x = len(df) - 12
    train = df.iloc[:x]
    test = df.iloc[x:]
    scaler = MinMaxScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    n_input = 12
    n_features = 1
    pred_list = []
    batch = train[-n_input:].reshape((1, n_input, n_features))
    for i in range(n_input):
        pred_list.append(model.predict(batch)[0])
        batch = np.append(batch[:, 1:, :], [[pred_list[i]]], axis=1)

    add_dates = [df.index[-1] + DateOffset(months=x) for x in range(0, 13)]
    future_dates = pd.DataFrame(index=add_dates[1:], columns=df.columns)
    df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                              index=future_dates[-n_input:].index, columns=['Prediction'])

    df_proj = pd.concat([df, df_predict], axis=1)
    df_proj = df_proj.round(0)
    last_twelve = df_proj.tail(12)
    col_predict_list = last_twelve['Prediction'].tolist()
    return col_predict_list


if __name__ == '__main__':
    app.run()
    # app.run(host='0.0.0.0', port=5000)
