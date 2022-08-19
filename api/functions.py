import json
import numpy as np
import cv2
import pandas as pd
from .apps import ApiConfig
from PIL import Image
import base64
import io


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


def lcs(a, b):
    a = str(a) if not isinstance(a,str) else a
    b = str(b) if not isinstance(b,str) else b
    prev = [0]*len(a)
    for i,r in enumerate(a):
        current = []
        for j,c in enumerate(b):
            if r==c:
                e = prev[j-1]+1 if i* j > 0 else 1
            else:
                e = max(prev[j] if i > 0 else 0, current[-1] if j > 0 else 0)
            current.append(e)
        prev = current
    return current[-1]

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
            scr = levenshtein(check_word,str(b))
            if scr < temp:
                temp = scr
                class_ = b
        return class_
    except Exception as e :
        print('region 추출 오류 발생.', region, check_word, '에러코드:', e)

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
            
    
def address_process(text):
    try:
        text = ' '.join(text).strip() if not isinstance(text, str) else text.strip()
        raw_text = text
#         print('주소 전체:',raw_text)
        region_check_fc = lambda x, y : y.find(x) if x in y else 999

        # 1. 시도명 추출 -> 주소 속 도, 시 키워드 찾는데 못찾으면 일단 space로 나눠 진행
        rg1_min, rg1_max = region_min_max(region = '시도명')
        region_1_idx = min(region_check_fc('도',text[rg1_min:rg1_max+1]),region_check_fc('시',text[rg1_min:rg1_max+1]))
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

def idcard_classification(img_shape, data):
    '''
    input
    [
    [[(90, 40), (352, 96), '주민등록증'],
     [(76, 111), (332, 161), '동길등 (람조)'],
     [(70, 174), (190, 204), '020701'],
     ]
     output
     {name:'홍길동',id_number:'~',}
    '''
    
    result = {'else':[],
            'else2':[],
            'name':[],
           'id_number':[],
           'address':[],
           'issue_date':[],
           'issuer':[]}
    
    temp_result = {'else':[],
            'else2':[],
            'name':[],
           'id_number':[],
           'address1':[],
           'address2':[],
           'address3':[],
           'address4':[],
           'issue_date':[],
           'issuer':[]}
    
    x_full, y = img_shape[1], img_shape[0]
    x = int(x_full * 0.8)
    
    form = {'else':[(0,0),(x,int(y*0.28))],
            'else2':[(x,0),(x_full,y)],
            'name':[(0,int(y*0.28)),(x,int(y*0.38))],
           'id_number':[(0,int(y*0.38)),(x,int(y*0.5))],
           'address1':[(0,int(y*0.5)),(x,int(y*0.57))],
           'address2':[(0,int(y*0.57)),(x,int(y*0.63))],
           'address3':[(0,int(y*0.63)),(x,int(y*0.69))],
           'address4':[(0,int(y*0.69)),(x,int(y*0.76))],
           'issue_date':[(0,int(y*0.76)),(x,int(y*0.84))],
           'issuer':[(0,int(y*0.84)),(x,y)]}
    
    for v in data:
        bbox1 = v[:2]
        temp_iou = 0
        class_ = ''
        for k in form:
            temp = iou(bbox1, form[k])
            if temp > temp_iou:
                temp_iou = temp
                class_ = k
        temp_result[class_].append(v[2])
        
#     print(temp_result)
    
    for i in temp_result:
        if 'address' in i:
            result['address'].append((' '.join(temp_result[i])))
        elif i == 'issuer':
            result[i].append(' '.join(temp_result[i]))
        else:
            result[i].append(''.join(temp_result[i]))
    
    
#     result['address'] = ' '.join(result['address']).strip()
    result['address'] = address_process(result['address'])
    result['name'] = name_process(result['name'])
    result['id_number'] = idnum_process(result['id_number'])
    result['issue_date'] = issue_date_precess(result['issue_date'])
    
    return result

def blur_image(img_data, res):
    '''
    img_data = 이미지 numpy배열
    '''
    dst = cv2.medianBlur(img_data, 77)

    for i in res:
        dst[i[0][1]+1:i[1][1]-1,i[0][0]+1:i[1][0]-1] = img_data[i[0][1]+1:i[1][1]-1,i[0][0]+1:i[1][0]-1]
        
    return dst

def result_look(img_data, reader, eng_reader):
    '''
    parameters
    img_data : 이미지 경로
    reader : easyOCR reader - 미리 load해야 함
    start_ratio - [x,y] 자르는거 시작점
    end_ratio - [x,y] 자르는거 끝지점
    thr - thr이하는 red로 표시
    '''
    result_ = []
    result1 = reader.readtext(img_data)
    result2 = eng_reader.readtext(img_data)
    img = img_data.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = [a if a[2] > b[2] else b for a,b in zip(result1,result2)]
    
    print('평균 confidence score:', np.mean([float(i[2]) for i in result1]),'->', np.mean([float(i[2]) for i in result]))

    for i in result: 
        x = i[0][0][0] 
        y = i[0][0][1] 
        w = i[0][1][0] - i[0][0][0] 
        h = i[0][2][1] - i[0][1][1]
        result_.append([(x, y), (x+w, y+h),str(i[1])])
    return idcard_classification(img.shape, result_), blur_image(np.array(img),result_), result_