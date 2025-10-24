from fastapi import APIRouter

api = APIRouter(prefix="/api")


@api.get("/hello")
def hello():
    return {"message": "Hello, Anon!"}