
import streamlit as st
import pandas as pd
import requests
import os

# Flask backend URL
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://backend:5000"
)

st.set_page_config(
    page_title="SuperKart Sales Prediction",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 SuperKart Sales Prediction")

st.write(
    "Predict product sales using product and store information."
)

tab1, tab2 = st.tabs([
    "Single Prediction",
    "Batch Prediction"
])


# --------------------------------------------------
# SINGLE PREDICTION
# --------------------------------------------------

with tab1:

    st.subheader("Single Sales Prediction")

    col1, col2 = st.columns(2)

    with col1:

        product_id = st.text_input(
            "Product ID",
            value="FDX01"
        )

        product_weight = st.number_input(
            "Product Weight",
            min_value=0.0,
            value=12.5
        )

        sugar_content = st.selectbox(
            "Product Sugar Content",
            ["Low Sugar", "Regular", "No Sugar"]
        )

        allocated_area = st.number_input(
            "Product Allocated Area",
            min_value=0.0,
            max_value=1.0,
            value=0.07
        )

        product_type = st.selectbox(
            "Product Type",
            [
                "Meat",
                "Snack Foods",
                "Hard Drinks",
                "Dairy",
                "Canned",
                "Soft Drinks",
                "Health and Hygiene",
                "Baking Goods",
                "Bread",
                "Breakfast",
                "Frozen Foods",
                "Fruits and Vegetables",
                "Household",
                "Seafood",
                "Starchy Foods",
                "Others"
            ]
        )

        product_mrp = st.number_input(
            "Product MRP",
            min_value=0.0,
            value=150.0
        )


    with col2:

        store_id = st.text_input(
            "Store ID",
            value="OUT001"
        )

        establishment_year = st.number_input(
            "Store Establishment Year",
            min_value=1900,
            max_value=2009,
            value=1998,
            step=1
        )

        store_size = st.selectbox(
            "Store Size",
            ["Low", "Medium", "High"]
        )

        city_type = st.selectbox(
            "Store Location City Type",
            ["Tier 1", "Tier 2", "Tier 3"]
        )

        store_type = st.selectbox(
            "Store Type",
            [
                "Departmental Store",
                "Supermarket Type 1",
                "Supermarket Type 2",
                "Food Mart"
            ]
        )


    if st.button("Predict Sales"):

        payload = {
            "Product_Id": product_id,
            "Product_Weight": product_weight,
            "Product_Sugar_Content": sugar_content,
            "Product_Allocated_Area": allocated_area,
            "Product_Type": product_type,
            "Product_MRP": product_mrp,
            "Store_Id": store_id,
            "Store_Establishment_Year": establishment_year,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_type,
            "Store_Type": store_type
        }

        try:

            response = requests.post(
                f"{BACKEND_URL}/predict",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:

                prediction = response.json()[
                    "predicted_sales"
                ]

                st.success(
                    f"Predicted Sales: {prediction:,.2f}"
                )

            else:

                st.error(
                    f"Prediction failed: {response.text}"
                )

        except requests.exceptions.RequestException as e:

            st.error(
                f"Unable to connect to backend: {e}"
            )


# --------------------------------------------------
# BATCH PREDICTION
# --------------------------------------------------

with tab2:

    st.subheader("Batch Sales Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        batch_data = pd.read_csv(uploaded_file)

        st.write("Uploaded Data")
        st.dataframe(batch_data.head())

        if st.button("Run Batch Prediction"):

            predictions = []
            errors = []

            progress = st.progress(0)

            total_rows = len(batch_data)

            for position, (_, row) in enumerate(
                batch_data.iterrows(), start=1
            ):

                # Target column is not sent to backend
                payload = row.drop(
                    labels=["Product_Store_Sales_Total"],
                    errors="ignore"
                ).to_dict()

                try:

                    response = requests.post(
                        f"{BACKEND_URL}/predict",
                        json=payload,
                        timeout=30
                    )

                    if response.status_code == 200:

                        predictions.append(
                            response.json()[
                                "predicted_sales"
                            ]
                        )

                        errors.append("")

                    else:

                        predictions.append(None)
                        errors.append(response.text)

                except requests.exceptions.RequestException as e:

                    predictions.append(None)
                    errors.append(str(e))

                progress.progress(
                    position / total_rows
                )

            result = batch_data.copy()

            result["Predicted_Sales"] = predictions

            if any(errors):
                result["Prediction_Error"] = errors

            st.success(
                "Batch prediction completed."
            )

            st.dataframe(result)

            csv = result.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Predictions",
                data=csv,
                file_name="superkart_predictions.csv",
                mime="text/csv"
            )
