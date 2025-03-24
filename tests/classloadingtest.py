class Dog:
    def __init__(self, name, breed):
        self.name = name
        self.breed = breed

    def speak(self):
        return f"{self.name} says: Woof!"


def create_instance(class_name, *args, **kwargs):
    cls = globals()[class_name]
    return cls(*args, **kwargs)


my_dog = create_instance("Dog", "Buddy", breed="Golden Retriever")
print(my_dog.speak())  # Buddy says: Woof!
print(globals())
