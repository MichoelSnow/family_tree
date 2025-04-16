from flask import Flask, request, jsonify, send_from_directory
import json
import os

app = Flask(__name__, static_folder='visualizations')

# Serve the main HTML file
@app.route('/')
def index():
    return send_from_directory('visualizations/main-viz', 'index.html')

# Serve all other static files from visualizations directory
@app.route('/visualizations/<path:path>')
def serve_static(path):
    return send_from_directory('visualizations', path)

# Save data endpoint
@app.route('/save-data', methods=['POST'])
def save_data():
    try:

        data = request.json
        # id_val = data["id"]
        # with open('visualizations/main-viz/data.json', 'r') as fp:
        #     data_in = json.load(fp)
        # data_out  = [x for x in data_in if x["id"]!= id_val]
        # data_out.append(data)
        # with open('visualizations/main-viz/data.json', 'w') as f:
        #     json.dump(data_out, f, indent=2)
        with open('visualizations/main-viz/data.json', 'w') as f:
            json.dump(data, f, indent=2)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)