import json
import os
from fastapi import APIRouter, File, UploadFile
from topics.NhanDangKhuonMatFacebook import Face_Recognition as face_reg
from PIL import Image

router = APIRouter()

def load_image(image_file):
	img = Image.open(image_file)
	return img

@router.post("/face-recognition/predict-name")
async def getSample(my_file: UploadFile = File(...)):

    data = ''
    
    try:
        with open(os.path.join("topics/NhanDangKhuonMatFacebook/images","test.bmp"),"wb+") as f:
            f.write(my_file.file.read())

        data = ''
        result = face_reg.onRecognition("topics/NhanDangKhuonMatFacebook/images/test.bmp")

        if(result == 'BanNinh'):
            data = "Bạn Ninh"

        elif(result == "BanThanh"):
            data = "Bạn Thành"

        elif(result == "ThayDuc"):
            data = "Thầy Đức"
        else: 
            data = "Not Recognite"
    except:
        data = "Not Size Image Standard (320x320)"

    
    result_data = {
        "data" : data
    }

    return result_data