# Testing

## What is covered

- dataset validation
- preprocessing behavior
- model file loading
- Flask health endpoint
- prediction endpoint validation
- invalid request handling

## Test command

```bash
cd ML\ Project\Thunderbird350X_Project
.venv311\Scripts\python.exe -m pytest
```

## Recommended test cases

1. CSV loads successfully.
2. Required dataset columns exist.
3. Preprocessor creates feature matrix correctly.
4. Model file exists and loads.
5. Health endpoint responds with 200.
6. Predict endpoint accepts valid input.
7. Invalid payload triggers a 400 response.

## Environment note

Because of the Windows Application Control policy affecting scipy/pandas DLLs, direct local full ML execution is blocked on this machine. Production testing should be done in a non-blocked environment such as Colab or a Linux VM.
