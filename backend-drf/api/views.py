from django.shortcuts import render
from rest_framework.views import APIView
from .serializers import StockPredictionSerializer
from rest_framework import status
from rest_framework.response import Response
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from django.conf import settings
from .utils import save_plot
from sklearn.preprocessing import MinMaxScaler
import traceback

# Import Keras/TensorFlow with error handling
try:
    from keras.models import load_model
except ImportError:
    from tensorflow.keras.models import load_model

from sklearn.metrics import mean_squared_error, r2_score

class StockPredictionAPIView(APIView):
    def post(self, request):
        try:
            serializer = StockPredictionSerializer(data=request.data)
            if serializer.is_valid():
                ticker = serializer.validated_data['ticker']

                # Debug: Print received ticker
                print(f"Received ticker: {ticker}")

                # Fetch the data from yfinance
                now = datetime.now()
                start = datetime(now.year-10, now.month, now.day)
                end = now
                
                print("Fetching data from yfinance...")
                df = yf.download(ticker, start, end)
                
                if df.empty:
                    return Response({
                        "error": "No data found for the given ticker.",
                        'status': 'error'
                    }, status=status.HTTP_404_NOT_FOUND)
                
                df = df.reset_index()
                print(f"Data fetched successfully. Shape: {df.shape}")

                # Ensure media directory exists
                media_dir = os.path.join(settings.BASE_DIR, 'media')
                os.makedirs(media_dir, exist_ok=True)

                # Generate Basic Plot
                plt.switch_backend('AGG')
                plt.figure(figsize=(12, 5))
                plt.plot(df.Close, label='Closing Price')
                plt.title(f'Closing price of {ticker}')
                plt.xlabel('Days')
                plt.ylabel('Price')
                plt.legend()
                plot_img_path = f'{ticker}_plot.png'
                plot_img = save_plot(plot_img_path)
                
                # 100 Days moving average
                ma100 = df.Close.rolling(100).mean()
                plt.figure(figsize=(12, 5))
                plt.plot(df.Close, label='Closing Price')
                plt.plot(ma100, 'r', label='100 DMA')
                plt.title(f'100 Days Moving Average of {ticker}')
                plt.xlabel('Days')
                plt.ylabel('Price')
                plt.legend()
                plot_img_path = f'{ticker}_100_dma.png'
                plot_100_dma = save_plot(plot_img_path)

                # 200 Days moving average
                ma200 = df.Close.rolling(200).mean()
                plt.figure(figsize=(12, 5))
                plt.plot(df.Close, label='Closing Price')
                plt.plot(ma100, 'r', label='100 DMA')
                plt.plot(ma200, 'g', label='200 DMA')
                plt.title(f'200 Days Moving Average of {ticker}')
                plt.xlabel('Days')
                plt.ylabel('Price')
                plt.legend()
                plot_img_path = f'{ticker}_200_dma.png'
                plot_200_dma = save_plot(plot_img_path)

                # Splitting data into Training & Testing datasets
                data_training = pd.DataFrame(df.Close[0:int(len(df)*0.7)])
                data_testing = pd.DataFrame(df.Close[int(len(df)*0.7): int(len(df))])

                # Scaling down the data between 0 and 1
                scaler = MinMaxScaler(feature_range=(0,1))

                # Check if model file exists
                model_path = os.path.join(settings.BASE_DIR, 'stock_prediction_model.keras')
                if not os.path.exists(model_path):
                    return Response({
                        "error": f"Model file not found at {model_path}",
                        'status': 'error'
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                # Load ML Model
                print("Loading model...")
                model = load_model(model_path)

                # Preparing Test Data
                past_100_days = data_training.tail(100)
                final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
                input_data = scaler.fit_transform(final_df)

                x_test = []
                y_test = []
                for i in range(100, input_data.shape[0]):
                    x_test.append(input_data[i-100: i])
                    y_test.append(input_data[i, 0])
                x_test, y_test = np.array(x_test), np.array(y_test)

                # Making Predictions
                print("Making predictions...")
                y_predicted = model.predict(x_test)

                # Revert the scaled prices to original price
                y_predicted = scaler.inverse_transform(y_predicted.reshape(-1, 1)).flatten()
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

                # Plot the final prediction
                plt.figure(figsize=(12, 5))
                plt.plot(y_test, 'b', label='Original Price')
                plt.plot(y_predicted, 'r', label='Predicted Price')
                plt.title(f'Final Prediction for {ticker}')
                plt.xlabel('Days')
                plt.ylabel('Price')
                plt.legend()
                plot_img_path = f'{ticker}_final_prediction.png'
                plot_prediction = save_plot(plot_img_path)

                # Model Evaluation
                mse = mean_squared_error(y_test, y_predicted)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_predicted)

                return Response({
                    'status': 'success',
                    'plot_img': plot_img,
                    'plot_100_dma': plot_100_dma,
                    'plot_200_dma': plot_200_dma,
                    'plot_prediction': plot_prediction,
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'r2': float(r2)
                })
            else:
                return Response({
                    'error': 'Invalid serializer data',
                    'details': serializer.errors
                }, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            # Log the full error traceback
            error_traceback = traceback.format_exc()
            print("ERROR:", error_traceback)
            
            return Response({
                'error': str(e),
                'status': 'error',
                'traceback': error_traceback  # Remove this in production
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
