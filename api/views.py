import os
import json
import numpy as np
from PIL import Image
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from config import settings
from . import forms
from .apps import ApiConfig
from django.shortcuts import render

@csrf_exempt # form post 요청을 받을 때 csrf 토큰없이 요청할 수 있도록 처리.
def predict(request):
    # 요청파라미터 - text: request.POST, file: request.FILES
    form = forms.UploadForm(request.POST, request.FILES)
    if form.is_valid(): #요청파라미터 검증. True: 검증 성공, False: 검증 실패
        clean_data = form.cleaned_data #Form에서 직접 값을 조회할 수 없다. form.cleaned_data: 검증을 통과한 
                                       #요청파라미터들을 딕셔너리로 반환. 이 딕셔너리를 이용해 조회
        img_field  = clean_data['upimg'] #업로드된 파일을 조회
        print(img_field, type(img_field))
        print(img_field.image.width, img_field.image.height, img_field.image.format, img_field.name) #ImageField.name: 파일명
        
        image = Image.open(img_field) # 이미지 로딩
        image_resize = image.resize((150,150)) #모델 input shape- (150,150,3)
        image_arr = np.array(image_resize) #PIL 이미지 타입을 ndarray 변환
        image_arr = image_arr/255. # 정규화
        input_tensor = image_arr[np.newaxis, ...] #배치 축 추가 (150,150,3) => (1, 150,150,3)

        model = ApiConfig.model
        pred = model.predict(input_tensor) #출력층 activation: sigoid  0:cat, 1:dog  [[0.7]]
        cls = np.where(pred[0,0]<0.5, 'cat', 'dog') 

        # 파일 저장 - 이 작업을 하지 않으면 업로드된 이미지는 응답할때 삭제된다.
        save_path = os.path.join(settings.MEDIA_ROOT, img_field.name)
        print(save_path)
        image.save(save_path) #PIL Image객체.save(경로) : 이미지 저장.
        
        
        # dictionary -> JSON 변환시 numpy 타입은 변환이 안된다. str(), float()으로 타입변환
        result = {
                'result':str(cls),
                'pred':float(pred[0,0]),
                'img_url':r"/media/{}".format(img_field.name)  
                }
        
        result_str = json.dumps(result)   #Dictionary 를 JSON 문자열로 변환.
    
        return HttpResponse(result_str)
# 응답형식
"""
{
 'result': 'cat',
 'pred':0.001,
 'img_url':'/media/abc.jpg'
}
"""      



def predict2(request):
    # 요청파라미터 - text: request.POST, file: request.FILES
    form = forms.UploadForm(request.POST, request.FILES)
    if form.is_valid(): #요청파라미터 검증. True: 검증 성공, False: 검증 실패
        clean_data = form.cleaned_data #Form에서 직접 값을 조회할 수 없다. form.cleaned_data: 검증을 통과한 
                                       #요청파라미터들을 딕셔너리로 반환. 이 딕셔너리를 이용해 조회
        img_field  = clean_data['upimg'] #업로드된 파일을 조회
        print(img_field, type(img_field))
        print(img_field.image.width, img_field.image.height, img_field.image.format, img_field.name) #ImageField.name: 파일명
        
        image = Image.open(img_field) # 이미지 로딩
        image_resize = image.resize((150,150)) #모델 input shape- (150,150,3)
        image_arr = np.array(image_resize) #PIL 이미지 타입을 ndarray 변환
        image_arr = image_arr/255. # 정규화
        input_tensor = image_arr[np.newaxis, ...] #배치 축 추가 (150,150,3) => (1, 150,150,3)

        model = ApiConfig.model
        pred = model.predict(input_tensor) #출력층 activation: sigoid  0:cat, 1:dog  [[0.7]]
        cls = np.where(pred[0,0]<0.5, 'cat', 'dog') 

        # 파일 저장 - 이 작업을 하지 않으면 업로드된 이미지는 응답할때 삭제된다.
        save_path = os.path.join(settings.MEDIA_ROOT, img_field.name)
        print(save_path)
        image.save(save_path) #PIL Image객체.save(경로) : 이미지 저장.
        
        
        # dictionary -> JSON 변환시 numpy 타입은 변환이 안된다. str(), float()으로 타입변환
        result = {
                'result':str(cls),
                'pred':float(pred[0,0]),
                'img_url':r"/media/{}".format(img_field.name)  
                }
        return render(request, 'api/response2.html', result)
        
