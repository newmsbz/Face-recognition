import cv2 as cv
import time
import numpy as np
import json
from os import listdir, remove
from datetime import datetime
from keras.models import load_model
from enum import IntEnum, auto
from mtcnn import MTCNN
from face_aligner import FaceAligner

detector = MTCNN()
face_aligner = FaceAligner()
model = load_model('model/facenet_keras.h5')


def face_detect(img):
    faces = detector.detect_faces(img)  # face 검출
    print(faces, len(faces))
    return faces

    # if faces[0]['confidence'] <= 0.999:  # face가 하나도 검출되지 않았을 경우
    #     return False
    # else:
    #     return True

    # if not faces:  # face가 하나도 검출되지 않았을 경우
    #     return False
    # else:
    #     return True


def choose_best_frame(conf_list, frame_list):
    max_conf_index = conf_list.index(max(conf_list))
    return cv.cvtColor(frame_list[max_conf_index], cv.COLOR_BGR2RGB)


# face 정보 임베딩 함수
def face_embedding(face):
    face_pixels = face.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()  # 화소의 평균, 분산
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]


def save_embedding_data(path, embedding_data, align):   # 경로, 얼굴 임베딩 정보, 얼굴 정렬 프레임
    nick_name = input("저장할 얼굴 정보의 닉네임을 입력해주세요 : ")

    db = dict()
    db_names = list()
    prev_check = [False]

    for file in listdir(path):
        db_names.append(file)

    for file_name in db_names:
        if nick_name in file_name:
            prev_check[0] = True
            print(f"동일 닉네임의 얼굴 정보 {file_name}이 이미 존재합니다.")
            while True:
                continue_check = input(f"이전 얼굴 정보를 삭제하고 현재 정보를 덮어씌우려면 1, skip 하려면 0을 입력해주세요 : ")
                if continue_check is '1':
                    remove(path + file_name)    # 이전 파일 삭제
                    print(f"이전 얼굴 정보 {file_name}이 삭제되었습니다.")

                    new_filename = nick_name + '_' + str(datetime.today())
                    replace_text = [':', '.', '/', '*', '?', '"', '<', '>', '|']

                    # 저장 불가능한 특수문자를 '-'로 변환
                    for text in replace_text:
                        new_filename = new_filename.replace(text, '-')

                    db['nick_name'] = nick_name
                    db['embedding_data'] = embedding_data
                    db['frame'] = align.tolist()
                    with open(f'{path + new_filename}.json', 'w', encoding='utf-8') as make_file:
                        json.dump(db, make_file)
                    print(f'{nick_name}님의 얼굴 정보가 새로 저장되었습니다.')
                    time.sleep(3)
                    return 0
                elif continue_check is '0':
                    print("진행 중인 얼굴정보 저장 작업을 취소하고 이전 상태로 돌아갑니다.")
                    print('-------------------------------------------------------------------')
                    return -1
                else:
                    print("잘못된 입력입니다.")

    if prev_check[0] is False:
        print(f"기존에 저장된 얼굴정보가 없으므로 {nick_name}님의 정보를 새로 저장합니다.")
        filename = nick_name + '_' + str(datetime.today())
        replace_text = [':', '.', '/', '*', '?', '"', '<', '>', '|']

        # 저장 불가능한 특수문자를 '-'로 변환
        for text in replace_text:
            filename = filename.replace(text, '-')

        db['nick_name'] = nick_name
        db['embedding_data'] = embedding_data
        db['frame'] = align.tolist()
        with open(f'{path + filename}.json', 'w', encoding='utf-8') as make_file:
            json.dump(db, make_file)
        print(f'{nick_name}님의 얼굴 정보 저장이 완료되었습니다.')
        time.sleep(3)
        print('-------------------------------------------------------------------')
        return 0

    # return 0 : 저장완료 -> 종료
    # return -1 : 저장 skip -> 이전 상태로


def l2_normalize(x: np.ndarray):
    return x / np.linalg.norm(x)


# (R, G, B) 좌표계 상의 색상 tuple
blue_color, red_color, green_color, white_color, black_color = (255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 255), \
                                                               (0, 0, 0)
box_range = ((180, 140), (420, 380))  # 캠카메라 640 * 480 기준 box 영역
box_thickness = 2
temp_fps_list, temp_frame_list, temp_conf_list = list(), list(), list()

cap = cv.VideoCapture(0)
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 480)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 640)

print('width :%d, height : %d' % (cap.get(3), cap.get(4)))

prev_time = time.time()
current_frame = 0
while True:
    ret, frame = cap.read()  # Read 결과와 frame

    current_time = time.time()
    sec = current_time - prev_time
    prev_time = current_time
    current_fps = 1 / sec
    # fps_list.append(current_fps)

    if ret:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # MTCNN은 BGR이 아닌 RGB에서 잘 작동하므로 BGR -> RGB 변환
        box = frame[box_range[0][1] + box_thickness: box_range[1][1] - box_thickness,
              box_range[0][0] + box_thickness: box_range[1][0] - box_thickness]
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        faces = face_detect(box)

        # 1, 2번째 프레임은 FPS가 제대로 측정되지 않기에 얼굴 인식에서 제외
        if current_frame is 0 or current_frame is 1:
            frame = cv.rectangle(frame, box_range[0], box_range[1], green_color, box_thickness)
            pass
        else:
            # 검출된 얼굴이 1개이고 정확도가 99.9% 이상인 경우
            if len(faces) is 1 and faces[0]['confidence'] >= 0.99:
                temp_conf_list.append(faces[0]['confidence'])
                temp_frame_list.append(frame)
                temp_fps_list.append(current_fps)
                frame = cv.rectangle(frame, box_range[0], box_range[1], red_color, box_thickness)

                # 최근 3초 동안 얼굴 인식 정확도가 0.999 이상이면
                if len(temp_conf_list) >= int(3 * (sum(temp_fps_list) / len(temp_fps_list))):
                    print("얼굴 인식이 완료되었습니다.")
                    best_frame = choose_best_frame(temp_conf_list, temp_frame_list)
                    best_face = detector.detect_faces(best_frame)
                    best_align = face_aligner.align(best_frame, best_face[0]['keypoints']['left_eye'],
                                                    best_face[0]['keypoints']['right_eye'])
                    best_align = cv.cvtColor(best_align, cv.COLOR_RGB2BGR)
                    best_embedded = l2_normalize(face_embedding(best_align)).tolist()
                    check = save_embedding_data('face data/', best_embedded, best_align)

                    if check is 0:
                        exit(0)
                    elif check is -1:
                        temp_conf_list.clear()
                        temp_frame_list.clear()
                    # cv.imshow('best_align', best_align)
                    # cv.waitKey(0)
            else:
                frame = cv.rectangle(frame, box_range[0], box_range[1], green_color, box_thickness)
                temp_conf_list.clear()
                temp_frame_list.clear()
                # print(f"현재 사각형 영역에 인식된 얼굴의 개수가 {len(faces)}개 입니다.")

        cv.putText(frame, 'FPS : ' + str(round(current_fps, 2)), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, black_color, 3)

        cv.imshow('frame_color', frame)  # 컬러 화면 출력
        # cv.imshow('frame_gray', gray)    # Gray 화면 출력
        # cv.imshow('box', box)  # 박스영역 화면 출력
        current_frame += 1
        if cv.waitKey(1) == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
