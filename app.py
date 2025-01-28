from flask import Flask, request, jsonify
from model import answer

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/parse', methods=['GET','POST'])
def process_json():
    try:
        # Validate and retrieve the input query
        input_data = request.get_json()
        if "value" not in input_data:
            return jsonify({"error": "Missing 'value' key in request JSON"}), 400
        
        query = input_data["value"]
        print(f"Received query: {query}")

        # Get prediction from the LLM
        prediction = answer(query)

        # Return the prediction
        return jsonify({"output": prediction})

    except Exception as e:
        # Return any errors with traceback
        error_message = {"error": str(e)}
        return jsonify(error_message), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
