**Weather Forecasting with Prophet: Mumbai Dataset**

This repository contains code for analyzing and forecasting weather data for Mumbai using the Prophet forecasting tool. The code imports weather data from a CSV file, preprocesses it, visualizes temperature trends, builds a predictive model using Prophet, and evaluates the model's performance through cross-validation.

**Dependencies**

pandas

seaborn

matplotlib

plotly

Prophet

**Description**

1.The Mumbai.csv file contains historical weather data for Mumbai.

2.The Python script weather_forecasting.py imports the required libraries, reads the CSV file, preprocesses the data, visualizes temperature trends over time, builds a predictive model using Prophet, and evaluates the model's performance through cross-validation.

3.The script generates line plots to visualize maximum and minimum temperatures in Mumbai over time.

4.It prepares the data frame for Prophet modeling by dropping unwanted columns and renaming columns.

5.The script extracts year and month from the date column for further analysis.

6.It creates line plots to visualize temperature change over the years and months in Mumbai.

7.The script uses Prophet to create a predictive model and generates forecasts for future temperatures.

8.Finally, it performs cross-validation of the model to evaluate its accuracy using Mean Absolute Percentage Error (MAPE) metric.
