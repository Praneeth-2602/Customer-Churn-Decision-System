# Backend

This folder contains the FastAPI backend for the Churn Decision Intelligence project.

- `main.py` — API entrypoint exposing `/analyze` which accepts `dataset` and `customer` payloads.
- `test_backend.py` — simple integration test that sends requests to the API (includes `dataset`).
- `test_simulated_customers.py` — generates synthetic customers and calls `/analyze` with `dataset`.

The backend is dataset-agnostic and forwards requests to the ML layer based on the `dataset` value.
