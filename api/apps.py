from django.apps import AppConfig
from easyocr import *
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from dataset import AlignCollate, RawDataset2
from utils import AttnLabelConverter
from model.easyOCR import model_test
# App/apps.py/ApiConfig 클래스
# App에서 사용할 자원을 Application이 시작할때(서버 처음 실행되는 시점) 생성하는 작업을 하는 클래스
#   ApiConfig클래스는 Application이 시작할 때 한번 실행된다.
#   App내의 다른 모듈들이 사용할 데이터(자원)을 class변수에 대입해 놓고 사용할 수 있도록 한다.
#   여기서는 저장된 모델을 loading 하여 model class변수에 저장해 놓는다. 
#                           => 모델은 추론할 때 마다 loading할 필요 없이 한번만 loading하면 되므로 시작할 때 한번 읽어온다.
class opt():
    imgH = 32
    imgW = 100
    num_fiducial = 20
    input_channel = 1
    output_channel = 512
    hidden_size = 256
    num_class = 38
    batch_max_length = 25
    image_folder ='./demo/'
    workers = 40
    batch_size = 16
    rgb = False
    Transformation = 'TPS'
    FeatureExtraction = 'ResNet'
    SequenceModeling = 'BiLSTM'
    Prediction = 'Attn'
    character = '0123456789abcdefghijklmnopqrstuvwxyz'

class ApiConfig(AppConfig):
    name = 'api'
#     AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=False)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     eng_model = model_test.Model(opt)
    
#     converter = AttnLabelConverter(opt.character)
#     opt.num_class = len(converter.character)
#     eng_model = torch.nn.DataParallel(eng_model).to(device)
#     eng_model.load_state_dict(torch.load('./model/easyOCR/Attn.pth', map_location=device))
    
    eng_model = Reader(['en'], gpu = True, detector = False)
    
    model = Reader(['ko'], gpu=True,
                model_storage_directory='./model/easyOCR',
                user_network_directory='./model/easyOCR',
                recog_network='custom')
    
    names = ['부산광역시','충청북도','충청남도','대구광역시','대전광역시','강원도','광주광역시','경상북도','경상남도','경기도','인천광역시','제주특별자치도','전라북도','전라남도','세종특별자치시','서울특별시','울산광역시']
    add_dict = dict()
    for named in names:
        with open("./api/unique_address/"+named, "rb" ) as file:
            loaded_data = pickle.load(file)
            add_dict[named] = loaded_data
    



