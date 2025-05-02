from pydantic import BaseModel


class Nest(BaseModel):
    eggs: int
    tree: str


class Owl(BaseModel):
    name: str
    interest: str
    nest: Nest


nest = Nest(eggs=25, tree="oak")
owl = Owl(name="Edwige", interest="AI", nest=nest)

print(owl)

config = {"name": "Edwige", "interest": "AI", "nest": {"eggs": 26, "tree": "oak"}}

init_owl = Owl(**config)

print(init_owl)
