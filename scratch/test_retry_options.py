from google.genai import types

try:
    retry_options = types.HttpRetryOptions(attempts=1)
    print("HttpRetryOptions created successfully:", retry_options)
    
    http_options = types.HttpOptions(
        api_version="v1beta",
        retry_options=retry_options
    )
    print("HttpOptions created successfully:", http_options)
except Exception as e:
    print("Error:", e)
