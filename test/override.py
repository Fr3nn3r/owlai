import pydash

original_json = {
    "name": "Alice",
    "age": 25,
    "address": {
        "city": "New York",
        "zip": "10001"
    }
}

overrides = {
    "age": 30,
    "address": {"city": "Los Angeles"}
}

new_json = pydash.merge(original_json.copy(), overrides)

print(new_json)
