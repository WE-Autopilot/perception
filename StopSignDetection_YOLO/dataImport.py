from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()

rf = Roboflow(api_key=os.getenv("api_key"))
project = rf.workspace("weap-cv-team").project("stop-sign-zn1kw-gqecq")
version = project.version(3)
dataset = version.download("yolov11")