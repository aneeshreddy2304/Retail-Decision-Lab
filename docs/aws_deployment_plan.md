# AWS Deployment Plan for Retail Decision Lab

## Goal

Deploy Retail Decision Lab as a client-facing retail demand forecasting dashboard on AWS.

## Recommended First Deployment

Use Docker with AWS App Runner or Elastic Beanstalk.

## Current Architecture

    User Browser
        ↓
    Streamlit Dashboard
        ↓
    Backend Inference Module
        ↓
    best_demand_model.joblib
        ↓
    Forecast Output

## Future AWS Architecture

    User Browser
        ↓
    AWS App Runner / ECS Fargate
        ↓
    Streamlit App Container
        ↓
    Amazon S3 for uploaded client files and model artifacts
        ↓
    CloudWatch logs and monitoring

## Why AWS

- Demonstrates cloud deployment readiness.
- Supports a realistic client-facing ML product workflow.
- Enables future storage of uploaded client files in Amazon S3.
- Enables future model artifact storage outside Git.
- Supports future monitoring through Amazon CloudWatch.

## Future Improvements

- Store uploaded client CSV files in S3.
- Store model artifacts in S3 instead of Git.
- Add CloudWatch logging.
- Add a scheduled retraining workflow.
- Add authentication for private client usage.
- Add an API service layer for programmatic forecasts.

## Resume Positioning

Built and deployed a client-ready retail demand forecasting dashboard using Python, scikit-learn, Streamlit, Docker, and AWS, with upload-based inference, persisted model serving, and executive analytics for demand planning.
