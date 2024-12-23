import numpy as np
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf
import requests
import plotly.graph_objs as go

# Télécharger et charger les données une seule fois
data = yf.download("AAPL", start="2010-01-01", end="2024-12-31")
data.to_csv("C:\\Users\\user\\Desktop\\apple_stock.csv")
data = pd.read_csv("C:\\Users\\user\\Desktop\\apple_stock.csv", skiprows=2)
data.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
data['Date'] = pd.to_datetime(data['Date'])
data = data.dropna()

# Modèle ARIMA
arima_model = ARIMA(data['Close'], order=(5, 1, 0))
arima_model_fit = arima_model.fit()

# Modèle GARCH
garch_model = arch_model(data['Close'], vol='Garch', p=1, q=1)
garch_model_fit = garch_model.fit()

# Régression linéaire
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Application Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=data['Date'].min(),
        end_date=data['Date'].max(),
        display_format='YYYY-MM-DD'
    ),
    dcc.Graph(id='price-graph'),
    dcc.Graph(id='real-time-graph'),
    dcc.Interval(
        id='interval-component',
        interval=5 * 60 * 1000,  # 5 minutes en millisecondes
        n_intervals=0
    )
])

@app.callback(
    Output('price-graph', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graph(start_date, end_date):
    filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    fig = px.line(filtered_data, x='Date', y='Close', title='Prix de l\'action Apple')
    return fig

@app.callback(
    Output('real-time-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_real_time_graph(n_intervals):
    # Remplacez cette partie par des données réelles si possible
    real_time_data = pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=10, freq='H'),
        'Value': np.random.randint(10, 20, size=10)
    })
    figure = {
        'data': [
            go.Scatter(
                x=real_time_data['Date'],
                y=real_time_data['Value'],
                mode='lines+markers',
                name='Value Over Time'
            )
        ],
        'layout': go.Layout(
            title='Graphique en Temps Réel',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Valeur'},
            showlegend=True
        )
    }
    return figure

if __name__ == "__main__":
    from waitress import serve
    # Lancer l'application avec Waitress pour le développement local
    serve(app.server, host="0.0.0.0", port=8000)
