from flask import Flask, render_template, request, jsonify
from predictor import StockPredictor
import traceback

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data    = request.get_json(force=True)
        ticker  = data.get('ticker', 'AAPL').strip().upper()
        period  = data.get('period', '2y')
        horizon = int(data.get('horizon', 5))

        if not ticker:
            return jsonify({'error': 'Ticker symbol is required'}), 400
        if horizon not in (1, 3, 5, 10, 20):
            return jsonify({'error': 'Horizon must be 1, 3, 5, 10, or 20'}), 400

        predictor = StockPredictor(ticker, period, horizon)
        result    = predictor.run()
        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(debug=debug, host='0.0.0.0', port=port)
