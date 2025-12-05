from roboflow import Roboflow
rf = Roboflow(api_key="BxgB1ceiWhT1t9Sjdyt2")
project = rf.workspace("weap-cv-team").project("stop-sign-zn1kw-gqecq")
version = project.version(3)
dataset = version.download("yolov11")