# Importation des bibliothèques
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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from dash import Dash
# Télécharger les données boursières d'Apple via yFinance
data = yf.download("AAPL", start="2010-01-01", end="2024-12-01")
data.to_csv("C:\\Users\\user\\Desktop\\apple_stock.csv")  # Sauvegarder les données dans un fichier CSV
# Charger et prétraiter les données
data = pd.read_csv("C:\\Users\\user\\Desktop\\apple_stock.csv", skiprows=2)

# Affiche les premières lignes pour vérifier les colonnes
print(data.head())

# Si nécessaire, ajuster les colonnes
data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# Convertir la colonne Date en format datetime
data['Date'] = pd.to_datetime(data['Date'])

# Supprimer les valeurs manquantes
data = data.dropna()

print(data.head())
# Modèle ARIMA pour la prédiction des prix futurs
arima_model = ARIMA(data['Close'], order=(5,1,0))
arima_model_fit = arima_model.fit()
arima_forecast = arima_model_fit.forecast(steps=10)  # Prédiction des 10 prochains jours
print("ARIMA Forecast:", arima_forecast)
# Modèle GARCH pour la prévision de la volatilité
garch_model = arch_model(data['Close'], vol='Garch', p=1, q=1)
garch_model_fit = garch_model.fit()
garch_forecast_volatility = garch_model_fit.forecast(horizon=10)
print("GARCH Volatility Forecast:", garch_model_fit.forecast(horizon=10).variance[-1:])
# Modèle de régression linéaire pour prédire les prix en fonction de plusieurs variables
X = data[['Open', 'High', 'Low', 'Volume']]  # Variables indépendantes
y = data['Close']  # Variable cible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
mse = mean_squared_error(y_test, lr_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, lr_predictions)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
# Visualisation des prédictions de régression linéaire
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Valeurs Réelles')
plt.plot(y_test.index, lr_predictions, label='Prédictions', linestyle='--')
plt.title('Régression Linéaire - Prédictions vs Réelles')
plt.xlabel('Date')
plt.ylabel('Prix de l\'action')
plt.legend()
plt.show()
# Création du graphique initial avec Plotly
fig = px.line(data, x='Date', y='Close', title='Prix de l\'action Apple')

# Application Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    # Sélecteur de plage de dates
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=data['Date'].min(),
        end_date=data['Date'].max(),
        display_format='YYYY-MM-DD'
    ),
    # Graphique pour afficher les prix
    dcc.Graph(id='price-graph'),
    # Nouveau graphique pour afficher la volatilité
    dcc.Graph(id='volatility-graph'),
    # Graphique en temps réel
    dcc.Graph(id='real-time-graph'),
    dcc.Interval(
        id='interval-component',
        interval=5 * 60 * 1000,  # 5 minutes en millisecondes
        n_intervals=0
    )
])
# Callback pour mettre à jour le graphique en fonction de la plage de dates sélectionnée
@app.callback(
    Output('price-graph', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graph(start_date, end_date):
    filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    fig = px.line(filtered_data, x='Date', y='Close', title='Prix de l\'action Apple')
    return fig
# Fonction pour obtenir des données en temps réel
def get_real_time_figure():
    try:
        api_key = 'votre_cle_alpha_vantage'  # Remplacez par votre clé API Alpha Vantage
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=5min&apikey={api_key}"
        response = requests.get(url)
        response.raise_for_status()  # Vérifie si la requête a réussi

        data = response.json()
        if "Time Series (5min)" in data:
            real_time_data = pd.DataFrame(data["Time Series (5min)"]).T
            real_time_data['Date'] = pd.to_datetime(real_time_data.index)
            real_time_data['Value'] = real_time_data['4. close'].astype(float)
        else:
            raise ValueError("Données manquantes dans la réponse de l'API")
    except (requests.exceptions.RequestException, ValueError) as e:
        # Si une erreur survient, utiliser des données fictives
        print(f"Erreur dans l'API: {e}")
        real_time_data = pd.DataFrame({
            'Date': pd.date_range(start='2024-01-01', periods=10, freq='H'),
            'Value': np.random.randint(10, 20, size=10)
        })

    # Créer la figure avec Plotly
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
# Callback pour mettre à jour les données en temps réel
@app.callback(
    dash.dependencies.Output('real-time-graph', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_realtime_data(n):
    return get_real_time_figure()
# Callback pour afficher la volatilité prédites avec GARCH
@app.callback(
    Output('volatility-graph', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_volatility_graph(n):
    garch_volatility = garch_model_fit.forecast(horizon=10).variance[-1:]
    dates = pd.date_range(start=data['Date'].max(), periods=10, freq='D')
    fig_volatility = go.Figure()
    fig_volatility.add_trace(go.Scatter(
        x=dates,
        y=garch_volatility.values.flatten(),
        mode='lines+markers',
        name='Volatilité Prédites'
    ))
    fig_volatility.update_layout(title='Volatilité Prédites avec GARCH',
                                  xaxis={'title': 'Date'},
                                  yaxis={'title': 'Volatilité'},
                                  showlegend=True)
    return fig_volatility
# Lancer l'application
if __name__ == '__main__':
    # Utilise le port 10000 fourni par Render
    port = int(os.environ.get("PORT", 10000))
    app.run_server(host="0.0.0.0", port=port)
    app.run_server(debug=True)
