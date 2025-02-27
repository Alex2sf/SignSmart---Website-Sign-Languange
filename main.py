import os
import random
import string
import uvicorn
import traceback
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
import pandas as pd


from fastapi import FastAPI, Response, UploadFile, WebSocket, Depends, Form, HTTPException, status, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, func, or_, Column, Integer, String, Time
from sqlalchemy.ext.declarative import declarative_base
from collections import Counter
from gtts import gTTS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from model import extract_keypoints, mediapipe_detection, get_model, mp_holistic, actions
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from passlib.context import CryptContext
from pydantic import BaseModel
from jose import jwt
# from fastapi_mail import FastMail, MessageSchema, ConnectionConfig

app = FastAPI()

# SQL Alchemy database setup
SQLALCHEMY_DATABASE_URL = "mysql+mysqldb://root@localhost/db_sign"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Video(Base):
    __tablename__ = 'videos'
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    file_path = Column(String, index=True)
    audio_path = Column(String, index=True) 

class TrainSchedule(Base):
    __tablename__ = "train_schedule"

    id = Column(Integer, primary_key=True, index=True)
    train_id = Column(String(50))
    ka_name = Column(String(100))
    station_id = Column(String(50))
    station_name = Column(String(100))
    time_est = Column(Time)
    
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(50), unique=True, index=True)
    hashed_password = Column(String)
    reset_token = Column(String, nullable=True)


Base.metadata.create_all(bind=engine)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = "mysecretkey"
ALGORITHM = "HS256"


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user_by_email_or_username(db: Session, email_or_username: str):
    return db.query(User).filter((User.email == email_or_username) | (User.username == email_or_username)).first()

def authenticate_user(db: Session, email_or_username: str, password: str):
    user = get_user_by_email_or_username(db, email_or_username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict):
    to_encode = data.copy()
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def predict_arrival_time(train_id, station_id, db: Session):
    # Memuat model dari file
    model_filename = 'linear_regression_model.pkl'
    model = joblib.load(model_filename)
    
    # Mengambil data dari database
    schedules = db.query(TrainSchedule).all()
    
    # Konversi data menjadi DataFrame
    data = pd.DataFrame([{
        'train_id': s.train_id,
        'station_id': s.station_id,
        'time_est': s.time_est
    } for s in schedules])

    # Mengonversi kolom 'time_est' menjadi datetime dan seconds
    data['time_est'] = pd.to_datetime(data['time_est'].astype(str))
    data['time_est_seconds'] = data['time_est'].view('int64') / 10**9

    # Encode 'train_id' dan 'station_id' sebagai categorical codes
    data['train_id_code'] = data['train_id'].astype('category').cat.codes
    data['station_id_code'] = data['station_id'].astype('category').cat.codes

    # Mendapatkan kode untuk train_id dan station_id yang diberikan
    train_id_code = data[data['train_id'] == train_id]['train_id_code'].iloc[0]
    station_id_code = data[data['station_id'] == station_id]['station_id_code'].iloc[0]

    # Prediksi waktu kedatangan berikutnya
    pred_time_seconds = model.predict([[train_id_code, station_id_code]])[0]
    pred_time = pd.to_datetime(pred_time_seconds, unit='s').time()  # Ambil hanya jam
    
    return pred_time.strftime('%H:%M')

# conf = ConnectionConfig(
#     MAIL_USERNAME="hanhans3000@gmail.com",
#     MAIL_PASSWORD="thnb ghmm jmrd lmsu",
#     MAIL_FROM="hanhans3000@gmail.com",
#     MAIL_PORT=587,
#     MAIL_SERVER="smtp.gmail.com",
#     MAIL_STARTTLS=True,  # Add this line if STARTTLS is required
#     MAIL_SSL_TLS=False,  # Add this line if SSL/TLS is required
#     USE_CREDENTIALS=True
# )
# Directory to store uploaded videos and audio
UPLOAD_DIR = "./static"
VIDEO_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "videos")
AUDIO_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "audio")

# Ensure the upload directories exist
os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")


class TrainPredictionRequest(BaseModel):
    train_id: str
    station_id: str

class Token(BaseModel):
    access_token: str
    token_type: str
    

@app.post("/token", response_model=Token)
def login_for_access_token(response: RedirectResponse, db: Session = Depends(get_db), email_or_username: str = Form(...), password: str = Form(...)):
    user = authenticate_user(db, email_or_username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email , username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    response = RedirectResponse(url="/translate_video_form", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="Authorization", value=f"Bearer {access_token}", httponly=True)
    return response

@app.get("/logout")
def logout(response: RedirectResponse):
    response = RedirectResponse(url="/")
    response.delete_cookie("Authorization")
    return response

@app.get("/", response_class=HTMLResponse)
def login_form():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SignSmart</title>
        <link rel="stylesheet" href="/static/home.css">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="navs">
            <nav class="navbar navbar-expand-lg custom-navbar">
                <a class="navbar-brand" href="/">
                    <img src="/static/lgoo.png" alt="Your Logo" height="50">
                </a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse justify-content-end" id="navbarNav">
                    <ul class="navbar-nav">
                        <li class="nav-item">
                            <a class="nav-link" href="/about">Tentang</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/realtime">Ai Terjemahan</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/search">Kamus</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/jadwal-kereta">Jadwal</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/prediksi-kedatangan-kereta">Prediksi Kedatangan Kereta</a>
                        </li>
                    </ul>
                </div>
            </nav>
        </div>

        <header class="header">
            <div class="header-content">
                <img src="/static/lgoo.png" alt="Your Logo" height="300px">
                <p>Sign Smart Menuju<br>Komunikasi Tanpa Batas</p>
            </div>
            <div class="header-form">
                <h2>Sign In</h2>
                <form action="/token" method="post">
                <div class="form-group">
                    <input type="text" class="form-control" placeholder="Enter Email or Username" name="email_or_username">
                </div>
                <div class="form-group">
                    <input type="password" class="form-control" placeholder="Password" name="password">
                </div>
                <button type="submit" class="btn">Sign In</button>
            </form>
            </div>
        </header>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
  
@app.get("/about")
def about():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Sign Smart</title>
      <link rel="stylesheet" href="/static/about.css">
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
         <div class="navs">
            <!-- Navbar -->
            <nav class="navbar navbar-expand-lg custom-navbar">
              <a class="navbar-brand" href="/">
                <img src="/static/lgoo.png" alt="Your Logo" height="50">
            </a>
              <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
              </button>
              <div class="collapse navbar-collapse justify-content-end" id="navbarNav">
                <ul class="navbar-nav">
                  <li class="nav-item">
                    <a class="nav-link" href="/about">Tentang</a>
                  </li>
                  <li class="nav-item">
                    <a class="nav-link" href="/realtime">Ai Terjemahan</a>
                  </li>
                  <li class="nav-item">
                    <a class="nav-link" href="/search">Kamus</a>
                  </li>
                  <li class="nav-item">
                    <a class="nav-link" href="/jadwal-kereta">Jadwal</a>
                  </li>
                  <li class="nav-item">
                                <a class="nav-link" href="/prediksi-kedatangan-kereta">Prediksi Kedatangan Kereta</a>
                </li>
                </ul>
              </div>
            </nav>
          </div>

    <!-- Intro Section -->
    <section class="intro-section">
      <div class="intro-text">
        <p>SignSmart adalah platform web inovatif yang menggunakan teknologi kecerdasan buatan (AI), khususnya model Long Short-Term Memory (LSTM), untuk menerjemahkan bahasa isyarat secara otomatis. LSTM adalah jenis jaringan saraf tiruan yang efektif dalam memproses dan mengingat urutan data yang panjang, seperti gerakan bahasa isyarat yang kompleks.
        </br> Kami meluncurkan platform ini di berbagai institusi, termasuk Kereta Api Indonesia (KAI), untuk membantu mengelola administrasi dan mendukung komunikasi antara petugas dan penumpang yang menggunakan bahasa isyarat</p>
      </div>
      <div class="intro-img">
        <img src="/static/train.png" alt="Train Image">
      </div>
    </section>

    <!-- JavaScript Libraries -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/prediksi-kedatangan-keretas")
async def prediksi_kedatangan_kereta(request: TrainPredictionRequest, db: Session = Depends(get_db)):
    try:
        # Query untuk memeriksa keberadaan train_id dan station_id
        train_schedule = db.query(TrainSchedule).filter_by(train_id=request.train_id, station_id=request.station_id).first()

        if train_schedule:
            predicted_time = predict_arrival_time(request.train_id, request.station_id, db)
            return {"train_id": request.train_id, "station_id": request.station_id, "predicted_time": predicted_time}
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ID kereta atau ID stasiun tidak ditemukan.")
    
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/prediksi-kedatangan-kereta", response_class=HTMLResponse)
async def read_index():
    html_content = """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediksi Kedatangan Kereta</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
      </head>
      <body>
          <div class="navs">
              <!-- Navbar -->
              <nav class="navbar navbar-expand-lg custom-navbar">
                <a class="navbar-brand" href="/">
                  <img src="/static/lgoo.png" alt="Your Logo" height="50">
              </a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                  <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse justify-content-end" id="navbarNav">
                  <ul class="navbar-nav">
                    <li class="nav-item">
                      <a class="nav-link" href="/about">Tentang</a>
                    </li>
                    <li class="nav-item">
                      <a class="nav-link" href="/realtime">Ai Terjemahan</a>
                    </li>
                    <li class="nav-item">
                      <a class="nav-link" href="/search">Kamus</a>
                    </li>
                    <li class="nav-item">
                      <a class="nav-link" href="/jadwal-kereta">Jadwal</a>
                    </li>
                    <li class="nav-item">
                                <a class="nav-link" href="/prediksi-kedatangan-kereta">Prediksi Kedatangan Kereta</a>
                    </li>
                  </ul>
                </div>
              </nav>
            </div>
    <div class="container mt-5">
        <h1>Prediksi Kedatangan Kereta</h1>
        <form id="predictionForm">
            <div class="mb-3">
                <label for="train_id" class="form-label">Train ID</label>
                <input type="text" class="form-control" id="train_id" required>
            </div>
            <div class="mb-3">
                <label for="station_id" class="form-label">Station ID</label>
                <input type="text" class="form-control" id="station_id" required>
            </div>
            <button type="submit" class="btn btn-primary">Submit</button>
        </form>
        <div class="mt-3" id="predictionResult"></div>
    </div>

   <script>
    document.getElementById('predictionForm').addEventListener('submit', async function(event) {
        event.preventDefault();
        const train_id = document.getElementById('train_id').value;
        const station_id = document.getElementById('station_id').value;

        const response = await fetch('/prediksi-kedatangan-keretas', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ train_id, station_id })
        });

        const result = await response.json();
        const resultDiv = document.getElementById('predictionResult');
        if (response.ok) {
            resultDiv.innerHTML = `<div class="alert alert-success">
                Train ID: ${result.train_id}<br>
                Station ID: ${result.station_id}<br>
                Predicted Arrival Time: ${result.predicted_time}
            </div>`;
        } else {
            resultDiv.innerHTML = `<div class="alert alert-danger">${result.detail}</div>`;
        }
    });
</script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.get("/translate_video_form", response_class=HTMLResponse)
def translate_video_form(request: Request):
    is_logged_in = request.cookies.get("Authorization") is not None
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Video and Audio Upload</title>
        <link rel="stylesheet" href="static/upload.css">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="navs">
            <nav class="navbar navbar-expand-lg custom-navbar">
                <a class="navbar-brand" href="#">
                    <img src="static/lgoo.png" alt="Your Logo" height="50">
                </a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse justify-content-end" id="navbarNav">
                    <ul class="navbar-nav">
                        <li class="nav-item">
                            <a class="nav-link" href="/about">Tentang</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/realtime">Ai Terjemahan</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/search">Kamus</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/jadwal-kereta">Jadwal</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/prediksi-kedatangan-kereta">Prediksi Kedatangan Kereta</a>
                        </li>
                        {"<li class='nav-item'><a class='nav-link' href='/logout'>Logout</a></li>" if is_logged_in else ""}
                    </ul>
                </div>
            </nav>
        </div>

        <section class="upload">
            <div class="form-group upload-box">
                <img src="static/upload.png" alt="Upload Icon" style="width: 50px; height: 50px;">
                <p>Upload Video Here</p>
                <form id="uploadForm" class="upload-form mt-3" enctype="multipart/form-data" action="/translate_video" method="post">
                    <input type="file" class="form-control" id="video" name="video" accept="video/mp4" required>
                    <input type="text" class="form-control mt-3" id="title" name="title" placeholder="Enter video title" required>
                    <button type="submit" class="btn btn-primary mt-3">Upload</button>
                </form>
            </div>
        </section>

        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.6/dist/umd/popper.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# Endpoint function
@app.post("/translate_video", response_class=HTMLResponse)
async def translate_video(
    response: Response, 
    title: str = Form(...), 
    video: UploadFile = Form(...), 
    db: Session = Depends(get_db)
):
    try:
        if video.content_type != "video/mp4":
            response.status_code = 400
            return {"error": "File is not a video!"}

        random_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + ".mp4"
        temp_video_path = os.path.join(VIDEO_UPLOAD_DIR, random_filename)

        with open(temp_video_path, "wb") as temp_video_file:
            temp_video_file.write(await video.read())

        video_db = Video(title=title, file_path=random_filename)
        db.add(video_db)
        db.commit()
        db.refresh(video_db)

        permanent_filename = f"{video_db.id}.mp4"
        permanent_video_path = os.path.join(VIDEO_UPLOAD_DIR, permanent_filename)
        os.rename(temp_video_path, permanent_video_path)

        video_db.file_path = f"/static/videos/{permanent_filename}"
        
        audio_filename = f"{video_db.id}.mp3"
        audio_path = os.path.join(AUDIO_UPLOAD_DIR, audio_filename)
        tts = gTTS(text=title, lang='en')
        tts.save(audio_path)

        video_db.audio_path = f"/static/audio/{audio_filename}"
        db.commit()

        processed_info = extract_useful_info(title)

        return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Website with Bootstrap</title>
          <link rel="stylesheet" href="static/uploads.css">
          <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
             <div class="navs">
            <nav class="navbar navbar-expand-lg custom-navbar">
                <a class="navbar-brand" href="#">
                    <img src="static/lgoo.png" alt="Your Logo" height="50">
                </a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse justify-content-end" id="navbarNav">
                    <ul class="navbar-nav">
                        <li class="nav-item">
                            <a class="nav-link" href="/about">Tentang</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/realtime">Ai Terjemahan</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/search">Kamus</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/jadwal-kereta">Jadwal</a>
                        </li>
                        <li class="nav-item">
                                <a class="nav-link" href="/prediksi-kedatangan-kereta">Prediksi Kedatangan Kereta</a>
                            </li>
                    </ul>
                </div>
            </nav>
        </div>

          <!-- Result Section -->
          <section class="result">
            <!-- Video Display -->
        <div class="result-box">
          <video controls autoplay width="100%" height="auto">
            <source src="{video_db.file_path}" type="video/mp4">
            Your browser does not support the video tag.
          </video>
        </div>

            <div class="result-info">
              <label for="title">Title:</label>
              <input type="text" id="title" value="{video_db.title}" readonly>
              <label for="voice">Voice:</label>
              <audio controls>
                <source src="{video_db.audio_path}" type="audio/mpeg">
                Your browser does not support the audio element.
              </audio>
              <label for="keywords">Keywords:</label>
              <input type="text" id="keywords" value="{', '.join(processed_info['keywords'])}" readonly>
            </div>
          </section>

          <!-- Result Buttons -->
          <div class="result-buttons">
            <a href="/translate_video_form">Upload Another Video</a>
            <a href="/search">Search Video</a>
            <a href="/">Home</a>
          </div>

          <!-- JavaScript Libraries -->
          <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """)
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return {"error": "Internal Server Error"}
    

@app.get("/jadwal-kereta", response_class=HTMLResponse)
def jadwal_kereta():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pencarian Jadwal Kereta Api KRL Commuter</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        .schedule-item {
            cursor: pointer;
        }
        .schedule-item:hover {
            background-color: #f0f0f0;
        }
        .modal-body p {
            margin: 0;
        }
        .route-item {
            border-bottom: 1px solid #ddd;
            padding: 10px 0;
        }
    </style>
</head>
      <body>
          <div class="navs">
              <!-- Navbar -->
              <nav class="navbar navbar-expand-lg custom-navbar">
                <a class="navbar-brand" href="/">
                  <img src="/static/lgoo.png" alt="Your Logo" height="50">
              </a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                  <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse justify-content-end" id="navbarNav">
                  <ul class="navbar-nav">
                    <li class="nav-item">
                      <a class="nav-link" href="/about">Tentang</a>
                    </li>
                    <li class="nav-item">
                      <a class="nav-link" href="/realtime">Ai Terjemahan</a>
                    </li>
                    <li class="nav-item">
                      <a class="nav-link" href="/search">Kamus</a>
                    </li>
                    <li class="nav-item">
                      <a class="nav-link" href="/jadwal-kereta">Jadwal</a>
                    </li>
                    <li class="nav-item">
                                <a class="nav-link" href="/prediksi-kedatangan-kereta">Prediksi Kedatangan Kereta</a>
                    </li>
                  </ul>
                </div>
              </nav>
            </div>
    <div class="container">
        <h1 class="mb-4">Pencarian Jadwal Kereta Api KRL Commuter</h1>
        <div class="mb-3">
            <label for="station" class="form-label">Stasiun:</label>
            <select id="station" class="form-control">
                <option value="">Pilih Stasiun</option>
                <!-- Options will be dynamically populated via JavaScript -->
            </select>
        </div>
        <div class="mb-3">
            <label for="timeFrom" class="form-label">Waktu Dari:</label>
            <input type="time" id="timeFrom" class="form-control">
        </div>
        <div class="mb-3">
            <label for="timeTo" class="form-label">Waktu Hingga:</label>
            <input type="time" id="timeTo" class="form-control">
        </div>
        <div class="mb-3">
            <button onclick="searchSchedules()" class="btn btn-primary">Cari</button>
        </div>
        <div class="schedule-list" id="scheduleList">
            <!-- Schedules will be dynamically populated via JavaScript -->
        </div>
    </div>

    <!-- Modal for train details -->
    <div class="modal fade" id="trainDetailModal" tabindex="-1" aria-labelledby="trainDetailModalLabel" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="trainDetailModalLabel">Detail Rute Kereta</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <p><strong>Nama Kereta:</strong> <span id="trainName"></span></p>
                    <p><strong>ID Kereta:</strong> <span id="trainId"></span></p>
                    <div id="routeList"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const apiUrl = 'https://api-partner.krl.co.id/krlweb/v1';
        const stationSelect = document.getElementById('station');
        const timeFromInput = document.getElementById('timeFrom');
        const timeToInput = document.getElementById('timeTo');
        const scheduleList = document.getElementById('scheduleList');
        const trainNameSpan = document.getElementById('trainName');
        const trainIdSpan = document.getElementById('trainId');
        const routeList = document.getElementById('routeList');

        // Function to fetch station data from API
        async function fetchStations() {
            const response = await fetch(`${apiUrl}/krl-station`);
            const data = await response.json();
            return data.data;
        }

        // Function to populate station dropdown
        async function populateStations() {
            const stations = await fetchStations();
            stations.forEach(station => {
                const option = document.createElement('option');
                option.value = station.sta_id;
                option.textContent = station.sta_name;
                stationSelect.appendChild(option);
            });
        }

        // Function to search schedules
        async function searchSchedules() {
            const stationId = stationSelect.value;
            const timeFrom = timeFromInput.value;
            const timeTo = timeToInput.value;
            if (!stationId || !timeFrom || !timeTo) {
                alert('Harap isi semua field');
                return;
            }
            const response = await fetch(`${apiUrl}/schedule?stationid=${stationId}&timefrom=${timeFrom}&timeto=${timeTo}`);
            const data = await response.json();
            displaySchedules(data.data);
        }

        // Function to display schedules
        function displaySchedules(schedules) {
            scheduleList.innerHTML = '';
            schedules.forEach(schedule => {
                const scheduleItem = document.createElement('div');
                scheduleItem.classList.add('card', 'mb-3', 'schedule-item');
                scheduleItem.innerHTML = `
                    <div class="card-body">
                        <h5 class="card-title">${schedule.ka_name}</h5>
                        <p class="card-text"><strong>Rute:</strong> ${schedule.route_name}</p>
                        <p class="card-text"><strong>Destinasi:</strong> ${schedule.dest}</p>
                        <p class="card-text"><strong>Waktu Estimasi:</strong> ${schedule.time_est}</p>
                        <p class="card-text"><strong>Waktu Tiba:</strong> ${schedule.dest_time}</p>
                    </div>
                `;
                scheduleItem.onclick = () => showTrainDetails(schedule.train_id);
                scheduleList.appendChild(scheduleItem);
            });
        }

        // Function to show train details
        async function showTrainDetails(trainId) {
            const response = await fetch(`${apiUrl}/schedule-train?trainid=${trainId}`);
            const data = await response.json();
            const trainDetails = data.data[0]; // Assuming single train detail is returned
            trainNameSpan.textContent = trainDetails.ka_name;
            trainIdSpan.textContent = trainDetails.train_id;
            routeList.innerHTML = '';

            data.data.forEach(route => {
                const routeItem = document.createElement('div');
                routeItem.classList.add('route-item');
                routeItem.innerHTML = `
                    <p><strong>Stasiun:</strong> ${route.station_name}</p>
                    <p><strong>Waktu Estimasi:</strong> ${route.time_est}</p>
                `;
                routeList.appendChild(routeItem);
            });

            // Show the modal
            const trainDetailModal = new bootstrap.Modal(document.getElementById('trainDetailModal'));
            trainDetailModal.show();
        }

        // Populate stations dropdown on page load
        populateStations();
    </script>

    <!-- Bootstrap JS and dependencies -->
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.6/dist/umd/popper.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.min.js"></script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.get("/realtime", response_class=HTMLResponse)
def get_realtime_page():
    html_content = """
   <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-time Sign Language Detection</title>
    <link rel="stylesheet" href="static/realtime.css">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="navs">
        <nav class="navbar navbar-expand-lg custom-navbar">
            <a class="navbar-brand" href="#">
                <img src="static/lgoo.png" alt="Your Logo" height="50">
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse justify-content-end" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="/about">Tentang</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/realtime">Ai Terjemahan</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/search">Kamus</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/jadwal-kereta">Jadwal</a>
                    </li>
                    <li class="nav-item">
                                <a class="nav-link" href="/prediksi-kedatangan-kereta">Prediksi Kedatangan Kereta</a>
                            </li>
                </ul>
            </div>
        </nav>
    </div>
    <section class="live-section">
        <div class="live-container">
            <div class="live-indicator">
                <div class="dot"></div>
                <span>live</span>
            </div>
            <video id="video" autoplay></video>
            <p id="result" class="h4"></p>
        </div>
    </section>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const video = document.getElementById('video');
        const result = document.getElementById('result');

        const constraints = {
            video: {
                width: { ideal: 640, max: 1280 },
                height: { ideal: 480, max: 720 },
                facingMode: "user"
            }
        };

        navigator.mediaDevices.getUserMedia(constraints)
            .then(stream => {
                video.srcObject = stream;
            })
            .catch(err => {
                console.error("Error accessing camera: " + err);
            });

        const ws = new WebSocket('ws://' + window.location.host + '/ws');

        ws.onopen = () => {
            console.log('WebSocket connection opened');
        };

        ws.onmessage = (event) => {
            console.log('Message from server ', event.data);
            result.innerText = event.data;
        };

        function sendFrame() {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(blob => {
                ws.send(blob);
            }, 'image/jpeg');
        }

        video.addEventListener('play', () => {
            setInterval(sendFrame, 100);
        });
    </script>
</body>
</html>

    """
    return html_content


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()
    model = get_model()
    sequence = []
    predictions = []
    window_size = 10  # Size of the rolling window
    consistency_threshold = 0.5  

    # Use MediaPipe holistic model for pose, face, and hand detection
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        try:
            while True:
                data = await websocket.receive_bytes()
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                image, results = mediapipe_detection(frame, holistic)
                
                # Check if the hand landmarks are valid
                if results.left_hand_landmarks is not None or results.right_hand_landmarks is not None:
                    keypoints = extract_keypoints(results)
                    
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    if len(sequence) == 30:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        predicted_index = np.argmax(res)
                        confidence = res[predicted_index]

                        if confidence > 0.5:  # Adjust confidence threshold as needed
                            action = actions[predicted_index]
                            predictions.append(action)
                            predictions = predictions[-window_size:]

                            # Check for consistency within the window
                            if predictions.count(action) / window_size >= consistency_threshold:
                                await websocket.send_text(f"{action}")
                                predictions = []  # Clear predictions to avoid repeated sends
                        else:
                            await websocket.send_text("False Negative")
        except Exception as e:
            traceback.print_exc()
        finally:
            await websocket.close()


# Serve static files
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

@app.get("/search", response_class=HTMLResponse)
def search_page():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pencarian Video</title>
        <!-- Bootstrap CSS -->
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="static/search.css">
      </head>
      <body>
          <div class="navs">
              <!-- Navbar -->
              <nav class="navbar navbar-expand-lg custom-navbar">
                <a class="navbar-brand" href="/">
                  <img src="/static/lgoo.png" alt="Your Logo" height="50">
              </a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                  <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse justify-content-end" id="navbarNav">
                  <ul class="navbar-nav">
                    <li class="nav-item">
                      <a class="nav-link" href="/about">Tentang</a>
                    </li>
                    <li class="nav-item">
                      <a class="nav-link" href="/realtime">Ai Terjemahan</a>
                    </li>
                    <li class="nav-item">
                      <a class="nav-link" href="/search">Kamus</a>
                    </li>
                    <li class="nav-item">
                      <a class="nav-link" href="/jadwal-kereta">Jadwal</a>
                    </li>
                    <li class="nav-item">
                    <a class="nav-link" href="/prediksi-kedatangan-kereta">Prediksi Kedatangan Kereta</a>
                    </li>
                  </ul>
                </div>
              </nav>
            </div>
            
        <div class="container">
            <h1 class="mt-5 text-center">Pencarian Video</h1>
            <form id="searchForm" class="mt-3">
                <div class="input-group">
                    <input type="text" id="searchQuery" class="form-control" placeholder="Masukkan kata kunci untuk pencarian" required>
                    <button type="submit" class="btn btn-outline-primary">Cari</button>
                </div>
            </form>
            <div id="searchResults" class="mt-4"></div>
        </div>


        <!-- Bootstrap JS Bundle with Popper -->
        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.6/dist/umd/popper.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.min.js"></script>
        <script>
            const form = document.getElementById('searchForm');
            const searchResults = document.getElementById('searchResults');

            form.addEventListener('submit', async (event) => {
                event.preventDefault();
                const query = document.getElementById('searchQuery').value;
                const response = await fetch(`/search_videos?query=${encodeURIComponent(query)}`);
                const data = await response.json();

                searchResults.innerHTML = '';
                data.results.forEach(video => {
                    const videoCard = document.createElement('div');
                    videoCard.classList.add('card', 'video-card');
                    videoCard.innerHTML = `
                        <div class="card-body">
                            <h5 class="card-title">${video.title}</h5>
                            <video width="320" height="240" controls>
                                <source src="${video.file_path}" type="video/mp4">
                                Your browser does not support the video tag.
                            </video>
                            <audio controls>
                                <source src="${video.audio_path}" type="audio/mpeg">
                                Your browser does not support the audio element.
                            </audio>
                        </div>
                    `;
                    searchResults.appendChild(videoCard);
                });
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/search_videos", response_class=JSONResponse)
async def search_videos(query: str, db: Session = Depends(get_db)):
    keywords = query.split()
    results = db.query(Video).filter(
        or_(*[func.lower(Video.title).contains(keyword.lower()) for keyword in keywords])
    ).all()
    return {
        "results": [
            {
                "title": video.title,
                "file_path": video.file_path,
                "audio_path": video.audio_path  # Include audio path
            } for video in results
        ]
    }

    
def extract_useful_info(text: str) -> dict:
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_words = [word for word in word_tokens if word.lower() not in stop_words and word.isalnum()]
    keywords = Counter(filtered_words).most_common(5)
    return {'keywords': [word for word, _ in keywords]}

def generate_audio_for_title(title: str, video_id: int) -> str:
    tts = gTTS(title)
    audio_path = os.path.join(AUDIO_UPLOAD_DIR, f"{video_id}.mp3")
    tts.save(audio_path)
    return f"/static/audio/{video_id}.mp3"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
