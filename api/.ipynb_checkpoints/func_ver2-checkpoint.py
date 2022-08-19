import json
import numpy as np
import cv2
import pandas as pd
from .apps import ApiConfig
from PIL import Image
import base64
import torch
import torch.nn.functional as F
import io
from dataset import AlignCollate, RawDataset2
from utils import AttnLabelConverter
from .apps import opt, ApiConfig

def detect_postprocess(i):
    result = []
    for a in i[0][0]:
        result.append([(a[0],a[2]),(a[1],a[3])]) # [(x,y),(x+w,y+h)]
    return result

# 문자 좌표에 따라 class 분류

def bbox_classification2(img, detected, form = None):
    '''
    data_form : 원하는 class 및 비율, 모델 설정
    form = {'else':[(0,0),(1,0.28),'eng'], 
            'issue_date':[(0,0.76),(1,0.84),'kor'],
            'issuer':[(0,0.84),(1,1),'num']}
    class_type = doc : 
    '''
    if form == 'id':
        form = {'else':[(0,0),(1,0.28),False],
                'else2':[(0.7,0),(1,1),False],
                'name':[(0,0.28),(0.7,0.38),'kor'],
               'id_number':[(0,0.38),(0.7,0.5),'num'],
               'address1':[(0,0.5),(0.7,0.57),'kor'],
               'address2':[(0,0.57),(0.7,0.63),'kor'],
               'address3':[(0,0.63),(0.7,0.69),'kor'],
               'address4':[(0,0.69),(0.7,0.76),'kor'],
               'issue_date':[(0,0.76),(0.7,0.84),'date'],
               'issuer':[(0,0.84),(0.7,1),'kor']}
        
    if form == 'foreign':
        form = {'else':[(0,0),(1,1),False],
               'id_number':[(0.5,0.2),(0.76,0.3),'num'],
                'name':[(0.45,0.35),(0.95,0.5),'eng'],
                'country':[(0.45,0.5),(0.95,0.6),'eng'],
               'qualification':[(0.55,0.64),(0.85,0.72),'eng'],
               'issue_date':[(0.75,0.76),(0.9,0.80),'date']}
        
    data_form = dict()
    result = {'eng':[],'kor':[]}
    model_type = dict()
    shape = img.shape[1], img.shape[0]
    mul_func = lambda x, y : (int(x[0]*y[0]),int(x[1]*y[1]))
    
    for temp in form:
        temp_data = form[temp]
        data_form[temp] = [mul_func(temp_data[0],shape),mul_func(temp_data[1],shape)]
        model_type[temp] = temp_data[2]
        
    for bbox in detected:
        temp_iou = 0
        class_ = ''
        for k in data_form:
            temp = iou(bbox, data_form[k])
            if temp > temp_iou:
                temp_iou = temp
                class_ = k
                
        if class_ == 'kor':
             result['kor'].append([bbox,class_])
        elif 'else' not in class_:
            result['eng'].append([image_crop(img,bbox),class_])
    return result, model_type
            
            
def bbox_classification(img, detected, form = None):
    '''
    data_form : 원하는 class 및 비율, 모델 설정
    form = {'else':[(0,0),(1,0.28),'eng'], 
            'issue_date':[(0,0.76),(1,0.84),'kor'],
            'issuer':[(0,0.84),(1,1),'num']}
    class_type = doc : 
    '''
    if form == 'id':
        form = {'else':[(0,0),(1,0.28),False],
                'else2':[(0.7,0),(1,1),False],
                'name':[(0,0.28),(0.7,0.38),'kor'],
               'id_number':[(0,0.38),(0.7,0.5),'num'],
               'address1':[(0,0.5),(0.7,0.57),'kor'],
               'address2':[(0,0.57),(0.7,0.63),'kor'],
               'address3':[(0,0.63),(0.7,0.69),'kor'],
               'address4':[(0,0.69),(0.7,0.76),'kor'],
               'issue_date':[(0,0.76),(0.7,0.84),'date'],
               'issuer':[(0,0.84),(0.7,1),'kor']}
        
    if form == 'foreign':
        form = {'else':[(0,0),(1,1),False],
               'id_number':[(0.5,0.2),(0.76,0.3),'num'],
                'name':[(0.45,0.35),(0.95,0.5),'eng'],
                'country':[(0.45,0.5),(0.95,0.6),'eng'],
               'qualification':[(0.55,0.64),(0.85,0.72),'eng'],
               'issue_date':[(0.85,0.76),(0.9,0.80),'date']}
        
    data_form = dict()
    result = dict()
    model_type = dict()
    shape = img.shape[1], img.shape[0]
    mul_func = lambda x, y : (int(x[0]*y[0]),int(x[1]*y[1]))
    
    for temp in form:
        result[temp] = []
        temp_data = form[temp]
        data_form[temp] = [mul_func(temp_data[0],shape),mul_func(temp_data[1],shape)]
        model_type[temp] = temp_data[2]
        
    for bbox in detected:
        temp_iou = 0
        class_ = ''
        for k in data_form:
            temp = iou(bbox, data_form[k])
            if temp > temp_iou:
                temp_iou = temp
                class_ = k
        result[class_].append(bbox)
            
    return result, model_type

def image_crop(img, bbox):
    crop_image = img[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]]
    norm_image = cv2.normalize(crop_image, None, 0, 255, cv2.NORM_MINMAX)
    norm_image = cv2.cvtColor(norm_image, cv2.COLOR_RGB2GRAY)
    _,norm_image = cv2.threshold(norm_image, -1, 255,  cv2.THRESH_TRUNC | cv2.THRESH_OTSU)
    return norm_image

def data_recognition2(img, bbox_class, model_type, kor_reader, eng_reader):
    result = dict()
    for class_ in bbox_class:

        if class_ == 'kor':
            for bbox, data in bbox_class[class_]:
                if not result.get(data): 
                    result[data] = []
                norm_image = image_crop(img,bbox)
                result[data].append(kor_reader.recognize(norm_image)[0][1:])
        elif class_ == 'eng':
            device = ApiConfig.device
            converter = ApiConfig.converter
            demo_data = RawDataset2(root=bbox_class[class_], opt=opt)
            demo_loader = torch.utils.data.DataLoader(
                demo_data, batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=ApiConfig.AlignCollate_demo, pin_memory=True)

            with torch.no_grad():
                for image_tensors, image_path_list in demo_loader:
                    batch_size = image_tensors.size(0)
                    image = image_tensors.to(device)

                    length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
                    text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

                    preds = eng_reader(image, text_for_pred, is_train=False)

                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, length_for_pred)

                    preds_prob = F.softmax(preds, dim=2)
                    preds_max_prob, _ = preds_prob.max(dim=2)
                    for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                        if not result.get(img_name): 
                            result[img_name] = []
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]
                        confidence_score = float(pred_max_prob.cumprod(dim=0)[-1].detach().cpu().numpy())
                        result[img_name].append([pred,confidence_score])
    return result


def data_recognition(img, bbox_class, model_type, kor_reader, eng_reader):
    result = dict()
    for class_ in bbox_class:
        if not result.get(class_): 
            result[class_] = []
        for bbox in bbox_class[class_]:
            crop_image = img[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]]
            norm_image = cv2.normalize(crop_image, None, 0, 255, cv2.NORM_MINMAX)
            norm_image = cv2.cvtColor(norm_image, cv2.COLOR_RGB2GRAY)
            _, norm_image = cv2.threshold(norm_image, -1, 255,  cv2.THRESH_TRUNC | cv2.THRESH_OTSU) 
            
            if model_type[class_] == 'kor':
                result[class_].append(kor_reader.recognize(norm_image)[0][1:])
            elif model_type[class_]:
                result[class_].append(eng_reader.recognize(norm_image)[0][1:])

    return result

def recog_postprocess(result, class_result = None, need_address = True):
    pre_result = []
    
    if class_result == 'id':
        class_result = {'name':[],'id_number':[],'address':[],'issue_date':[],'issuer':[]}
        
        for i in result:
            if len(result[i]) > 0:
                if 'address' in i:
                    class_result['address'].append(' '.join([a[0] for a in result[i] if a[0].strip() != '']))
                elif i == 'issuer':
                    class_result[i].append(' '.join([a[0] for a in result[i] if a[0].strip() != '']))
                else:
                    class_result[i].append(''.join([a[0] for a in result[i] if a[0].strip() != '']))
                pre_result.append(''.join([a[0] for a in result[i] if a[0].strip() != '']))

        class_result['name'] = name_process(class_result['name'])
        class_result['id_number'] = idnum_process(class_result['id_number'])
        class_result['address'] = address_process(class_result['address']) if need_address else ''.join(class_result['address'])
        class_result['issue_date'] = issue_date_precess(class_result['issue_date'])
        class_result['issuer'] = class_result['issuer'][0]
        
    elif class_result == 'foreign':
        class_result = {'name':[],'id_number':[],'country':[],'issue_date':[],'qualification':[]}
        for i in result:
            if i == 'qualification':
                txt = ' '.join([a[0] for a in result[i] if a[0].strip() != ''])
                txt = '(' + txt.split('(')[-1]
                class_result[i].append(txt)
                pre_result.append(txt)
            elif len(result[i]) > 0:
                txt = ' '.join([a[0] for a in result[i] if a[0].strip() != ''])
                class_result[i].append(txt)
                pre_result.append(txt)
        
        print(class_result)
        for i in class_result:
            try:
                class_result[i] = class_result[i][0]
            except:
                pass
    
    return class_result, pre_result

def iou(box1, box2):
    # box = [(x1,y1),(x2,y2)]
    box1_area =(box1[1][0] - box1[0][0] + 1) * (box1[1][1] - box1[0][1] + 1)
    box2_area =(box2[1][0] - box2[0][0] + 1) * (box2[1][1] - box2[0][1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0][0], box2[0][0])
    y1 = max(box1[0][1], box2[0][1])
    x2 = min(box1[1][0], box2[1][0])
    y2 = min(box1[1][1], box2[1][1])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1    #왜냐면 matrix에 넣어줄때, 제일 왼쪽에 0이 붙으니까~~
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y)) # matrix를 0으로 초기화함
    for x in range(size_x): 
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):  #북서쪽 앞에서 채워준거 제외해줘야하니까 1부터 시작~
        for y in range(1, size_y):  #for, for로 비교
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,#문자삽입
                    matrix[x-1, y-1],   #문자제거 
                    matrix[x, y-1] + 1   #문자변경
                )
            else:
                matrix [x,y] = min(   #둘이 안같으면 그냥 전값에 1더해주기
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])  # matrix에서 제일 오른쪽아래값을 출력해줌 

# n-gram
def ngram(s, num): #
    res = []
    slen = len(s) - num + 1  # num개로 찢으면 몇개나오나 연산
    for i in range(slen): 
        ss = s[i:i+num]  #찢어준다!
        res.append(ss)   #res안에 조각조각낸거 넣어줌
    return res

def diff_ngram(sa, sb, num):   #sa,sb가 비교할 string, num은 몇개로 찢을거냐
    a = ngram(sa, num)  #sa를 num으로 찢은게 a
    b = ngram(sb, num)  #sb를 num으로 찢은게 b
    r = []
    cnt = 0
    for i in a:   #찢어준 리스트중에 머가 곂치는지 하나하나 비교해줌
        for j in b:
            if i == j:
                cnt += 1
                r.append(i) 
    return cnt / len(a)#곂치는 조각들을 모아서 리스트로 만들어줌


def region_min_max(region, pre_region = False, first_region = False):
    add_dict = ApiConfig.add_dict
    names = ['부산광역시','충청북도','충청남도','대구광역시','대전광역시','강원도','광주광역시','경상북도','경상남도','경기도','인천광역시','제주특별자치도','전라북도','전라남도','세종특별자치시','서울특별시','울산광역시']
    region_list = ['시도명','시군구명','도로명','건물본번']
    if region == '시도명':
        region = names
    elif region == '시군구명':
        region = add_dict[pre_region]['시군구명'].unique()
    else:
        region = add_dict[first_region][add_dict[first_region][region_list[region_list.index(region)-1]] == pre_region][region].unique()
    length = list(map(len,list(map(str, region))))
    return min(length)-1, max(length)-1

def region_check(region, check_word, pre_region = False, first_region = False):
    '''
    region -> 현재 찾을 곳의 범주(시도명, 시군구명, 도로명, 건물본번)
    check_word -> 검사할 단어
    pre_region -> 이전 범주의 결과값(없을 경우 시도명을 return)
    '''
    # 시도명 + 시군구명 + (읍면동명이 면 인 경우) + 도로명 + 건물 본번 dataset 생성 및 진행
    try:
        add_dict = ApiConfig.add_dict
        names = ['부산광역시','충청북도','충청남도','대구광역시','대전광역시','강원도','광주광역시','경상북도','경상남도','경기도','인천광역시','제주특별자치도','전라북도','전라남도','세종특별자치시','서울특별시','울산광역시']
        region_list = ['시도명','시군구명','도로명','건물본번']
        if region == '시도명':
            region = names
        elif region == '시군구명':
            region = add_dict[pre_region]['시군구명'].unique()
        else:
            region = add_dict[first_region][add_dict[first_region][region_list[region_list.index(region)-1]] == pre_region][region].unique()
        temp = 999
        class_ = ''
        for b in region:
            scr =  levenshtein(check_word,str(b)) - diff_ngram(check_word,str(b),1)
            if scr < temp:
                temp = scr
                class_ = b
        return class_
    except Exception as e :
        print('region 추출 오류 발생.', region, check_word, '에러코드:', e)
        
def address_process(text):
    try:
        text = ' '.join(text).strip() if not isinstance(text, str) else text.strip()
        raw_text = text
#         print('주소 전체:',raw_text)
        region_check_fc = lambda x, y : y.find(x) if x in y else 999

        # 1. 시도명 추출 -> 주소 속 도, 시 키워드 찾는데 못찾으면 일단 space로 나눠 진행
        rg1_min, rg1_max = region_min_max(region = '시도명')
        region_1_idx = region_check_fc('도',text[rg1_min:rg1_max+1])
        region_1_idx = region_check_fc('시',text[rg1_min:rg1_max+1]) if region_1_idx == 999 else region_1_idx
        region_1_idx = region_check_fc(' ',text[rg1_min:rg1_max+1]) if region_1_idx == 999 else region_1_idx
        region_1 = text[:region_1_idx + rg1_min + 1].strip()
        text = text[region_1_idx + rg1_min + 1:].lstrip()
        region_1_rs = region_check(region = '시도명',check_word = region_1)
        print('시도명:',region_1 , region_1_rs)

        # 2. 시군구명 추출 -> 수원시 영통구 같은 부분은 구나 군까지 추출해야 하기 때문에 먼저 추출, 최대 길이 4!!
        # 반례: 만약 도로명이나 뒤에 구나 군이 포함되면 이상하게 추출 
        # 해결방안
        #     1. 구나 군이 들어갈 수 있는 최대 길이를 확인 후 진행
        #     2. 구나 군 부분 추출한 것과 매칭 vs 시로 매칭한것과 매칭 점수를 비교하여 더 높은 쪽에 부여
        rg2_min, rg2_max = region_min_max(region = '시군구명', pre_region = region_1_rs)
        region_2_idx = min(region_check_fc('구',text[rg2_min:rg2_max+1]),region_check_fc('군',text[rg2_min:rg2_max+1]))
        region_2_idx = min(region_check_fc('시',text[rg2_min:rg2_max+1]),region_check_fc(' ',text[rg2_min:rg2_max+1])) if region_2_idx == 999 else region_2_idx
        region_2 = text[:region_2_idx+1+rg2_min].strip()
        text = text[region_2_idx+1+rg2_min:].lstrip()
        region_2_rs = region_check(region = '시군구명', check_word = region_2, pre_region = region_1_rs)

        # 3. 도로명 추출 -> 줄 넘김으로 인해 오탈자가 많이 발생하는 부분. 로와 길 중 인덱스가 더 높은 걸로 추출
        # 도로명 + 건물본번 -> 영동로 114 , 장안로26가길 105 -> 만약 길을 인식하지 못한다면 space와 숫자의 인덱스 중 더 작은 인덱스로 추출
        rg3_min, rg3_max = region_min_max(region = '도로명', pre_region = region_2_rs, first_region = region_1_rs)
        region_3_idx = text[rg3_min:rg3_max+1].find('길')
        region_3_idx = text[rg3_min:rg3_max+1].find('로') if region_3_idx == -1 else region_3_idx
        if region_3_idx == -1:
            region_3_idx = text[rg3_min:rg3_max+1].find(' ')
            region_3_idx = min(region_3_idx, hasNumber(text[rg3_min:rg3_max+1])-1) if hasNumber(text[rg3_min:rg3_max+1]) != False else region_3_idx
        region_3 = text[:region_3_idx+1+rg3_min].replace(" " , "")
        text = text[region_3_idx+1+rg3_min:].lstrip()
        region_3_rs = region_check(region = '도로명',check_word = region_3, pre_region = region_2_rs, first_region = region_1_rs)

        # 4. 건물본번 추출 -> 숫자로만 이뤄져 있고, 본번 뒤에는 상세주소가 붙어 띄어쓰기가 있음.
        rg4_min, rg4_max = region_min_max(region = '건물본번', pre_region = region_3_rs, first_region = region_1_rs)
        region_4_idx = text[rg4_min:rg4_max+1].find(' ') if text[rg4_min:rg4_max+1].find(' ') != -1 else rg4_max - rg4_min
        region_4 = text[:region_4_idx+1+rg4_min].strip()
        text = text[region_4_idx+1+rg4_min:].lstrip()
        region_4_rs = str(region_check(region = '건물본번',check_word = region_4, pre_region = region_3_rs, first_region = region_1_rs))

        result = ' '.join([region_1_rs, region_2_rs, region_3_rs, region_4_rs, text])
        print('시도명:',region_1,'->',region_1_rs,'\n시군구명:',region_2,'->',region_2_rs,'\n도로명:',region_3,'->',region_3_rs,'\n건물본번:',region_4,'->',region_4_rs,'\n나머지:',text)
        print('기존:',raw_text,'\n변환:',result)

        return result
    except Exception as e:
        print('address 후처리 실패', e)
        return raw_text
    
def name_process(text):
    try:
        text = ''.join(text).strip()
        idx = text.find('(') if '(' in text else 3
        print('이름:' + text + '->' , text[:idx])
        return text[:idx]
    except Exception as e:
        print('name 후처리 실패',e)
        return text
    
def idnum_process(text):
    try:
        text = ''.join(text).strip()
        result = ''
        cnt = 0
        for i in text:
            if cnt == 6:
                result = result + '-'
                cnt += 1
            if i.isdigit():
                result = result + i
                cnt += 1
        print('주민등록번호:' + text + '->' + result)
        return result
    except Exception as e:
        print('idnum 후처리 실패',e)
        return text

def issue_date_precess(text):
    try:
        text = ''.join(text).strip()
        result = ''
        cnt = 0
        for i in text:
            if i.isdigit():
                if cnt == 0:
                    if i in ['1','2']:
                        result = result + i
                        cnt += 1
                elif cnt == 1:
                    if i in ['0','9']:
                        result = result + i
                        cnt += 1
                else:
                    result = result + i
                    cnt += 1
        print('발행일자:' + text + '->' + result[:4] + '.' + result[4:-2] + '.'+result[-2:])
        return result[:4] + '.' + result[4:-2] + '.'+result[-2:]
    except Exception as e:
        print('issue_date 후처리 실패',e)
        return text
    
    
def blur_image(img_data, res):
    '''
    img_data = 이미지 numpy배열
    '''
    dst = cv2.medianBlur(img_data, 77)

    for i in res:
        dst[i[0][1]+1:i[1][1]-1,i[0][0]+1:i[1][0]-1] = img_data[i[0][1]+1:i[1][1]-1,i[0][0]+1:i[1][0]-1]
        
    return dst