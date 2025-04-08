USER_DATABASE = {
    "user_id_4385972043572": {
        "password": "red unicorn",
        "role": "admin",
        "last_name": "Brunner",
        "first_name": "Frederic",
        "date_of_birth": "1978-03-30",
        "phone": "+41788239217",
        "favorite": {
            "color": "pink",
            "animal": "owl",
            "food": "stuffed tomatoes",
            "drink": "monster",
            "movie": "The Matrix",
            "music": "Hard Rock",
            "book": "The Beginning of Infinity",
            "activity": "hiking, lifting weights, reading, coding",
        },
        "created": "2025-02-27",
        "updated": "2025-03-05",
    },
    "user_id_4385972043573": {
        "password": "pink dragon",
        "role": "user",
        "last_name": "Brunner",
        "first_name": "Luc",
        "date_of_birth": "2014-06-18",
        "phone": "None",
        "favorite": {
            "color": "pink",
            "animal": "dragon",
            "food": "lasagna",
            "drink": "cocacola",
            "movie": "Frozen",
            "music": "Rap",
            "book": "Naruto",
            "activity": "video games",
        },
        "created": "2025-02-27",
        "updated": "2025-02-27",
    },
    "user_id_4385972043574": {
        "password": "shiny flower",
        "role": "user",
        "last_name": "Brunner",
        "first_name": "Claire",
        "date_of_birth": "2020-12-17",
        "phone": "None",
        "favorite": {
            "color": "pink",
            "animal": "unicorn",
            "food": "chicken nuggets",
            "drink": "monster",
            "movie": "Frozen",
            "music": "Twinkle twinkle little star",
            "book": "Peppa pig",
            "activity": "coloring in",
        },
        "created": "2025-02-27",
        "updated": "2025-02-27",
    },
}


def get_user_by_password(password):
    for user_id, user_data in USER_DATABASE.items():
        if user_data["password"] == password:
            return {**user_data, "user_id": user_id}  # Include user_id in the response
    return None  # Return None if password is not found
