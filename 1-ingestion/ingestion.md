1. `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` means that the returned object is not valid JSON. Most likely due to invalid/expired `APP_ID` and/or `API_KEY` values. Possible solutions: 
    
    1. Check the output of the `APP_ID` and `API_KEY`. 
    2. Log `response.status_code/response.text` before calling `.json` to see the real error payload.