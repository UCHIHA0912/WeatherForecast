#importing required libraries, reading and describing csv
import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt
import plotly.express as px
df=pd.read_csv("/content/Mumbai.csv")
print(df.describe())
df.fillna(method='ffill', inplace=True)
df.tmax.fillna(method='bfill', inplace=True)
#Plotting the maximum temperatures 
figure = px.line(df, x="time", 
                 y="tmax", 
                 title='Maximum Temperatures in Mumbai')
figure.show()
#Plotting the minimum temperatures 
figure = px.line(df, x="time", 
                 y="tmin", 
                 title='Minimum Temperatures in Mumbai')
figure.show()
#dropping unwanted columns and preparing data frame for prophet modelling
df=df.drop(['tmax'],axis=1)
df=df.drop(['tmin'],axis=1)
df=df.drop(["prcp"],axis=1)
df.rename(columns = {'time':'ds','tavg':'y'}, inplace = True)
print(df.to_string())

#Extraction of year and month
df["ds"] = pd.to_datetime(df["ds"], format = '%d-%m-%Y')
df['year'] = df['ds'].dt.year
df["month"] = df["ds"].dt.month
#Plotting temperature over the years
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.title("Temperature Change in Mumbai Over the Years")
sns.lineplot(data = df, x='month', y='y', hue='year')
plt.show()
#plotting temperatures over the months
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.title("Temperature Change in Mumbai Over the Years")
sns.lineplot(data = df, x='year', y='y', hue='month')
plt.show()
#Creating a predictive model using prophet
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
model = Prophet()
model.fit(df)
forecasts = model.make_future_dataframe(periods=365)
predictions = model.predict(forecasts)
plot_plotly(model, predictions)
model.plot(predictions)
#cross validation of model
from prophet.diagnostics import cross_validation

df_cv = cross_validation(model, initial='730 days', period='180 days', horizon = '365 days')
from prophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
df_p.head()
from prophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='mape')
cutoffs = pd.to_datetime(['2013-02-15', '2013-08-15', '2014-02-15'])
df_cv2 = cross_validation(model, cutoffs=cutoffs, horizon='365 days')
