
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("superkart_model.joblib")


def prepare_input(data):
    """
    Performs the same feature engineering used during model training.
    """
    df = pd.DataFrame([data])

    # Extract product category from Product_Id
    df["Product_Category"] = df["Product_Id"].str[:2]

    # Create Store_Age using the same reference year used during training
    df["Store_Age"] = 2009 - df["Store_Establishment_Year"]

    # Remove columns that were excluded during model training
    df = df.drop(
        columns=[
            "Product_Id",
            "Store_Id",
            "Store_Establishment_Year"
        ]
    )

    return df


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "SuperKart Sales Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()

        processed_data = prepare_input(input_data)

        prediction = model.predict(processed_data)[0]

        return jsonify({
            "predicted_sales": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
