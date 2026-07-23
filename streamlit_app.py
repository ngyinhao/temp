import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource(show_spinner="Training the Random Forest model...")
def train_model():
    housing = fetch_california_housing(as_frame=True)
    features = housing.data

    # scikit-learn stores the target in units of $100,000.
    target_dollars = housing.target * 100_000

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target_dollars,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)
    metrics = {
        "RMSE": mean_squared_error(y_test, test_predictions) ** 0.5,
        "MAE": mean_absolute_error(y_test, test_predictions),
        "R²": r2_score(y_test, test_predictions),
    }

    feature_statistics = pd.DataFrame(
        {
            "minimum": features.min(),
            "maximum": features.max(),
            "median": features.median(),
        }
    )
    feature_importance = (
        pd.Series(model.feature_importances_, index=features.columns)
        .sort_values(ascending=False)
        .rename("Importance")
    )

    return model, metrics, feature_statistics, feature_importance


model, metrics, feature_statistics, feature_importance = train_model()

st.title("California Housing Price Predictor")
st.write(
    "Estimate the median house value for a California district using a "
    "Random Forest regression model."
)

metric_columns = st.columns(3)
metric_columns[0].metric("Test RMSE", f"${metrics['RMSE']:,.0f}")
metric_columns[1].metric("Test MAE", f"${metrics['MAE']:,.0f}")
metric_columns[2].metric("Test R²", f"{metrics['R²']:.3f}")

st.divider()

input_column, result_column = st.columns([1.15, 0.85], gap="large")

with input_column:
    st.subheader("Housing district features")
    st.caption("Defaults are the median values in the California housing data.")

    entered_values = {}
    feature_labels = {
        "MedInc": "Median income (tens of thousands of dollars)",
        "HouseAge": "Median house age (years)",
        "AveRooms": "Average rooms per household",
        "AveBedrms": "Average bedrooms per household",
        "Population": "District population",
        "AveOccup": "Average household occupancy",
        "Latitude": "Latitude",
        "Longitude": "Longitude",
    }

    with st.form("prediction_form"):
        form_columns = st.columns(2)
        for position, feature_name in enumerate(feature_statistics.index):
            statistics = feature_statistics.loc[feature_name]
            with form_columns[position % 2]:
                entered_values[feature_name] = st.number_input(
                    feature_labels[feature_name],
                    min_value=float(statistics["minimum"]),
                    max_value=float(statistics["maximum"]),
                    value=float(statistics["median"]),
                    format="%.4f",
                )

        submitted = st.form_submit_button(
            "Predict house value",
            type="primary",
            use_container_width=True,
        )

with result_column:
    st.subheader("Prediction")

    input_frame = pd.DataFrame(
        [[entered_values[name] for name in feature_statistics.index]],
        columns=feature_statistics.index,
    )
    predicted_value = model.predict(input_frame)[0]

    if submitted:
        st.success(f"Estimated median house value: **${predicted_value:,.0f}**")
    else:
        st.info("Adjust the inputs and select **Predict house value**.")

    st.subheader("Feature importance")
    st.bar_chart(feature_importance, horizontal=True)
    st.caption(
        "Importance measures each feature's contribution to reducing prediction "
        "error across the Random Forest."
    )

with st.expander("Model and data details"):
    st.markdown(
        """
        - **Dataset:** scikit-learn California Housing dataset
        - **Model:** Random Forest Regressor
        - **Trees:** 250
        - **Maximum tree depth:** 20
        - **Minimum samples per leaf:** 2
        - **Test split:** 20%
        - **Random seed:** 42

        Tree-based models split values at thresholds, so feature scaling is not
        required.
        """
    )
