from sqlalchemy.orm import Session
from main import User, engine, get_password_hash
from passlib.hash import bcrypt

db = Session(engine)

admin = User(
    username="Rezahans",
    email="rezacrent3@gmail.com",
    hashed_password=get_password_hash("Admin123")
)

db.add(admin)
db.commit()
db.close()
