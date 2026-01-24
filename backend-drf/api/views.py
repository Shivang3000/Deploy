from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from .serializers import StockPredictionSerializer
from rest_framework import status
from rest_framework.response import Response

# ML imports
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score

# helper function
from .utils import save_plot

class StockPredictionAPIView(APIView):
    def post(self, request):
        serializer = StockPredictionSerializer(data=request.data)

        if serializer.is_valid():
            ticker = serializer.validated_data['ticker']
            
            now = datetime.now()
            start = datetime(now.year-10, now.month, now.day)
            end = now

            df = yf.download(ticker, start, end)
            # print(df)

            if df.empty:
                return Response({'error':'No data found for this ticker',
                                 'status':status.HTTP_404_NOT_FOUND})
            
            df = df.reset_index()
            plt.switch_backend('AGG')
            plt.figure(figsize=(12,5))
            plt.plot(df.Close, 'b', label='Closing Price')
            plt.title(f'Closing Price of {ticker}')
            plt.xlabel('Days')
            plt.ylabel('Closing Price')
            plt.legend()

            #save the plot to a file
            plot_img_path = f'{ticker}_plot.png'
            plot_img = save_plot(plot_img_path)

            # 100 Days moving average
            ma100=df.Close.rolling(100).mean()
            plt.switch_backend('AGG')
            plt.figure(figsize=(12,5))
            plt.plot(df.Close, 'b', label='Closing Price')
            plt.plot(ma100, 'r', label='100 DMA')
            plt.title(f'100 DMA of {ticker}')
            plt.xlabel('Days')
            plt.ylabel('Closing Price')
            plt.legend()

            #save the plot to a file
            plot_img_path = f'{ticker}_100_dma.png'
            plot_100_dma = save_plot(plot_img_path)

            # 200 Days moving average
            ma200=df.Close.rolling(200).mean()
            plt.switch_backend('AGG')
            plt.figure(figsize=(12,5))
            plt.plot(df.Close, 'b', label='Closing Price')
            plt.plot(ma100, 'r', label='100 DMA')
            plt.plot(ma200, 'g', label='200 DMA')
            plt.title(f'200 DMA of {ticker}')
            plt.xlabel('Days')
            plt.ylabel('Closing Price')
            plt.legend()

            # save the plot to a file
            plot_img_path = f'{ticker}_200_dma.png'
            plot_200_dma = save_plot(plot_img_path)

            # splitting data into test & train
            data_training = pd.DataFrame(df.Close[0:int(len(df)*0.7)])
            data_testing = pd.DataFrame(df.Close[int(len(df)*0.7):len(df)])

            # scaling down the data between 0 & 1
            scaler = MinMaxScaler(feature_range=(0,1))

            # Load ML model
            model = load_model('stock_prediction_model.keras')

            # preparing test data
            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.fit_transform(final_df)

            # creating test case
            x_test = []
            y_test = []

            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100 : i])
                y_test.append(input_data[i, 0])
            
            x_test, y_test = np.array(x_test), np.array(y_test)

            # maping predictions
            y_predicted = model.predict(x_test)

            # Revert the scaled prices to the original price
            y_predicted = scaler.inverse_transform(y_predicted.reshape(-1,1)).flatten()
            y_test = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

            # ploting the final prediction
            plt.switch_backend('AGG')
            plt.figure(figsize=(12,5))
            plt.plot(y_test, 'r', label='Original Price')
            plt.plot(y_predicted, 'g', label='Pridicted Price')
            plt.title(f'Final Predicition for {ticker}')
            plt.xlabel('Days')
            plt.ylabel('Closing Price')
            plt.legend()

            # save the plot to a file
            plot_img_path = f'{ticker}_final_prediction.png'
            plot_prediction = save_plot(plot_img_path)

            # mean squared error
            mse = mean_squared_error(y_test, y_predicted)

            # root mean squared error
            rmse = np.sqrt(mse)

            # r-squared
            r2 = r2_score(y_test, y_predicted)

            return Response({'status':'success',
                             'ticker':ticker,
                             'plot_img':plot_img,
                             'plot_100_dma':plot_100_dma,
                             'plot_200_dma':plot_200_dma,
                             'plot_prediction':plot_prediction,
                             'mse':mse,
                             'rmse':rmse,
                             'r2':r2
                             })