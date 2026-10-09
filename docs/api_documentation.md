# API Documentation

The backend is a Flask REST API used to serve ML predictions and garage recommendations.

## Base URL

- Local: `http://127.0.0.1:5000`
- Android emulator: `http://10.0.2.2:5000`

## Endpoints

### GET /health

Returns status of the backend.

### POST /predict-availability

Input example:

```json
{
  "bike_model": "Royal Enfield Thunderbird 350X",
  "manufacturing_year": 2019,
  "part_name": "Clutch Cable",
  "part_category": "Control",
  "supplier_type": "Online Supplier",
  "city": "Pune",
  "part_condition": "New",
  "service_priority": "High"
}
```

Response:

```json
{
  "availability": "Available",
  "confidence": 0.82
}
```

### POST /predict-wait

Returns estimated waiting period in days.

### POST /predict-price

Returns estimated price in INR.

### POST /predict-part-details

Returns a combined summary using availability, wait, and price prediction.

### POST /find-garages

Returns ranked garage/supplier suggestions with a recommendation score.

### POST /service-request

Accepts a service request payload and stores it in JSON.

## Errors

- 400 for invalid input
- 500 for model or server failures
- no Python stack traces are exposed to clients
