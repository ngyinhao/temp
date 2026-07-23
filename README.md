# California Housing Random Forest App

A Streamlit application that predicts the median house value for a California district using a Random Forest regression model.

## Features

- Interactive inputs for all eight California Housing dataset features
- Random Forest prediction in US dollars
- Test RMSE, MAE, and R-squared metrics
- Feature-importance chart
- Cached model training for faster repeat visits

## Files

- streamlit_app.py — application, model training, evaluation, and prediction UI
- requirements.txt — Python dependencies for Streamlit Community Cloud

## Run locally

    python -m pip install -r requirements.txt
    streamlit run streamlit_app.py

The application downloads scikit-learn's public California Housing dataset on its first run, trains the model, and caches it for later sessions.

## Deploy

Create a new app at https://share.streamlit.io/, select this repository and the main branch, set the main file path to streamlit_app.py, and deploy.
