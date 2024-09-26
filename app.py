from flask import Flask, render_template, request, jsonify
from model import train_models, predict_demand
import pandas as pd
import plotly
import plotly.graph_objs as go
import json


app = Flask(__name__)

# Train models at startup
rf, lstm_model, xgb_model = train_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    holiday = int(request.form['holiday'])
    real_estate_growth = float(request.form['real_estate_growth'])
    timestamp = pd.to_datetime(request.form['timestamp'])

    input_data = {
        'temperature': temperature,
        'humidity': humidity,
        'holiday': holiday,
        'real_estate_growth': real_estate_growth,
        'timestamp': timestamp
    }

    # Make prediction
    prediction = predict_demand(rf, lstm_model, xgb_model, input_data)

    # Generate a plot for the predicted demand
    predicted_values = [prediction[0]]  # Current prediction
    timestamps = [timestamp]

    # Create a Plotly graph
    graph = go.Figure()

    # Add the predicted demand to the graph
    graph.add_trace(go.Scatter(x=timestamps, y=predicted_values, mode='lines+markers', name='Predicted Demand'))

    graph.update_layout(
        title='Predicted Electricity Demand Over Time',
        xaxis_title='Timestamp',
        yaxis_title='Predicted Demand (MW)'
    )

    # Convert the graph to JSON for rendering in the HTML template
    graph_json = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)

    # Prepare response
    return render_template('result.html', prediction=prediction[0], graph_json=graph_json)

if __name__ == '__main__':
    app.run(debug=True)
