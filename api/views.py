import os
import json
import numpy as np
import cv2
from time import time
from PIL import Image
from django.http import HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from config import settings
from . import forms, functions, func_ver2
from .apps import ApiConfig
from django.shortcuts import render
from django.http import FileResponse
from django.shortcuts import render
from .models import *
import random
import io
import base64

# ver.1
# @csrf_exempt # form post 요청을 받을 때 csrf 토큰없이 요청할 수 있도록 처리.
# def predict(request):
#     # 요청파라미터 - text: request.POST, file: request.FILES
#     form = forms.UploadForm(request.POST, request.FILES)
#     if form.is_valid(): #요청파라미터 검증. True: 검증 성공, False: 검증 실패
#         clean_data = form.cleaned_data #Form에서 직접 값을 조회할 수 없다. form.cleaned_data: 검증을 통과한 
#                                        #요청파라미터들을 딕셔너리로 반환. 이 딕셔너리를 이용해 조회
#         img_field  = clean_data['upimg'] #업로드된 파일을 조회
        
#         print(img_field, type(img_field))
#         print(img_field.image.width, img_field.image.height, img_field.image.format, img_field.name) #ImageField.name: 파일명

#         image = Image.open(img_field) # 이미지 로딩
#         image_arr = np.array(image)
#         model = ApiConfig.model
#         eng_model = ApiConfig.eng_model
#         result, blur_img, pre_result = functions.result_look(image_arr, model, eng_model)
        
#         img = Image.fromarray(cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB))
#         img = img.resize((410, 260))
#         blur_url = './img/img/blur'+str(random.random())+'.jpg'
#         img.save(blur_url)
#         result['url'] = blur_url[1:]
#         result['pre_result'] = [i[2] for i in pre_result]
        
#         return HttpResponse(json.dumps(result))

# ver.2
@csrf_exempt # form post 요청을 받을 때 csrf 토큰없이 요청할 수 있도록 처리.
def predict(request):
    start_time = time()
    # 요청파라미터 - text: request.POST, file: request.FILES
    form = forms.UploadForm(request.POST, request.FILES)
    if form.is_valid(): #요청파라미터 검증. True: 검증 성공, False: 검증 실패
        clean_data = form.cleaned_data #Form에서 직접 값을 조회할 수 없다. form.cleaned_data: 검증을 통과한 
                                       #요청파라미터들을 딕셔너리로 반환. 이 딕셔너리를 이용해 조회
        img_field  = clean_data['upimg'] #업로드된 파일을 조회
        print(img_field, type(img_field))
        print(img_field.image.width, img_field.image.height, img_field.image.format, img_field.name) #ImageField.name: 파일명
        form = 'id'
        image = Image.open(img_field) # 이미지 c로딩
        image_arr = np.array(image)
        
        kor_model = ApiConfig.model
        eng_model = ApiConfig.eng_model
        
        # img detect
        print('load 시간:', time() - start_time)
        start_time = time()
        
        detect_result = func_ver2.detect_postprocess(kor_model.detect(image_arr))
        print('detect 시간:', time() - start_time)
        start_time = time()
        
        # 데이터 분류 , form에 양식 넣기
        bbox_class, model_type = func_ver2.bbox_classification(image_arr, detect_result, form = form)
        
        # 글씨 분류
        rec_result = func_ver2.data_recognition(image_arr, bbox_class, model_type, kor_model, eng_model)
        print('reconize 시간:', time() - start_time)
        start_time = time()
        
        # 후처리 - 모델에 따라 customize 필요 
        result, pre_result = func_ver2.recog_postprocess(rec_result, need_address = True, class_result = form)
        print('후처리 시간:', time() - start_time)
        start_time = time()
        
        blur_img = func_ver2.blur_image(image_arr, detect_result)
        img = Image.fromarray(blur_img)
        img = img.resize((410, 260))
        blur_url = './img/img/blur'+str(random.random())+'.png'
        img.save(blur_url)
        result['url'] = blur_url[1:]
        result['pre_result'] = pre_result
        result['form'] = form
        
        return HttpResponse(json.dumps(result))

@csrf_exempt # form post 요청을 받을 때 csrf 토큰없이 요청할 수 있도록 처리.
def predict2(request):
    # 요청파라미터 - text: request.POST, file: request.FILES
    start_time = time()
    form = forms.UploadForm(request.POST, request.FILES)
    if form.is_valid(): #요청파라미터 검증. True: 검증 성공, False: 검증 실패
        clean_data = form.cleaned_data #Form에서 직접 값을 조회할 수 없다. form.cleaned_data: 검증을 통과한 
                                       #요청파라미터들을 딕셔너리로 반환. 이 딕셔너리를 이용해 조회
        img_field  = clean_data['upimg'] #업로드된 파일을 조회
        print(img_field, type(img_field))
        print(img_field.image.width, img_field.image.height, img_field.image.format, img_field.name) #ImageField.name: 파일명
        form = 'foreign'
        image = Image.open(img_field) # 이미지 c로딩
        image_arr = np.array(image)
#         image_arr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
        
        kor_model = ApiConfig.model
        eng_model = ApiConfig.eng_model
        print('load 시간:', time() - start_time)
        start_time = time()
        
        # img detect
        detect_result = func_ver2.detect_postprocess(kor_model.detect(image_arr))
        print('detect 시간:', time() - start_time)
        start_time = time()
        # 데이터 분류 , form에 양식 넣기
        bbox_class, model_type = func_ver2.bbox_classification(image_arr, detect_result, form = form)
        
        # 글씨 분류
        rec_result = func_ver2.data_recognition(image_arr, bbox_class, model_type, kor_model, eng_model)
        print('reconize 시간:', time() - start_time)
        start_time = time()
        
        # 후처리 - 모델에 따라 customize 필요 
        result, pre_result = func_ver2.recog_postprocess(rec_result, need_address = True, class_result = form)
        print('후처리 시간:', time() - start_time)
        start_time = time()
        
        blur_img = func_ver2.blur_image(image_arr, detect_result)
        img = Image.fromarray(blur_img)
        img = img.resize((410, 260))
        blur_url = './img/img/blur'+str(random.random())+'.png'
        img.save(blur_url)
        result['url'] = blur_url[1:]
        result['pre_result'] = pre_result
        result['form'] = form
        
        return HttpResponse(json.dumps(result))