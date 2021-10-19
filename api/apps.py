from django.apps import AppConfig
from tensorflow.keras import models
# App/apps.py/ApiConfig 클래스
# App에서 사용할 자원을 Application이 시작할때(서버 처음 실행되는 시점) 생성하는 작업을 하는 클래스
#   ApiConfig클래스는 Application이 시작할 때 한번 실행된다.
#   App내의 다른 모듈들이 사용할 데이터(자원)을 class변수에 대입해 놓고 사용할 수 있도록 한다.
#   여기서는 저장된 모델을 loading 하여 model class변수에 저장해 놓는다. 
#                           => 모델은 추론할 때 마다 loading할 필요 없이 한번만 loading하면 되므로 시작할 때 한번 읽어온다.
class ApiConfig(AppConfig):
    name = 'api'
    model = models.load_model(r'.\model\cat_dog_mobilenetv2')
    



