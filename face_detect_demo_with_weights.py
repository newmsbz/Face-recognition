import cv2 as cv
import numpy as np
import json
import time
import tensorflow as tf
from os import listdir
from keras.models import load_model
from enum import IntEnum, auto
from mtcnn import MTCNN
from face_aligner import FaceAligner


detector = MTCNN()
face_aligner = FaceAligner()
model = load_model('model/facenet_keras.h5')
db_path = 'face data/'


custom_model = tf.keras.models.load_model('ver2 128 64 32 16 (adam) batch dropout.h5')


# 저장된 얼굴 데이터 불러오기
def load_database(path):
    if len(listdir(path)) == 0:  # 저장된 얼굴정보가 하나도 없을 경우 0을 return
        print('저장된 얼굴정보가 하나도 없습니다. 얼굴정보 저장을 먼저 진행해주세요.')
        return 0
    db_list = list()
    for file in listdir(path):
        with open(path + file, 'r') as f:
            db: dict = json.load(f)
            for k in db:
                if k != 'nick_name':
                    db[k] = np.array(db[k], np.float32)
        db_list.append(db)
    print('저장된 얼굴정보를 성공적으로 불러왔습니다 !')
    return db_list


def face_detect(img):
    faces = detector.detect_faces(img)  # face 검출
    # print(faces, len(faces))
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


# L2 정규화
def l2_normalize(x: np.ndarray):
    return x / np.linalg.norm(x)


# 불러온 db와 캠카메라의 face를 비교하여 얼굴 detect
def compare_face_db(faces_db, cur_emb, cur_align):
    best = {'dist': 0}

    for face_db in faces_db:
        # print(np.array([cur_emb]))
        # print(face_db['embedding_data'], face_db['embedding_data'].shape)
        input = np.hstack([np.array([cur_emb]).ravel(), face_db['embedding_data']]).reshape(1, 256)
        predictions = custom_model.predict(input)
        print(face_db['nick_name'], predictions)
        if predictions[0][1] >= best['dist']:
            best['dist'], best['nick_name'], best['frame'] = predictions[0][1], face_db['nick_name'], face_db['frame']

    try:
        best_frame = best['frame'].astype(np.uint8)
        compare = cv.hconcat([cur_align, best_frame])
        print(f"비교 결과 : {best['nick_name']}")
        print(f"정확도 : {round(best['dist']*100, 2)}%")
        cv.imshow('Current & Detect result', compare)
        cv.waitKey(0)
    except KeyError as e:
        print("현재 얼굴과 유사한 얼굴 db가 없습니다 !!!")
        return 0
    # best = {'dist': 0}
    # for face_db in faces_db:
    #     dist = np.linalg.norm(cur_emb - face_db['embedding_data'])
    #     if dist < 0.9:
    #         if best['dist'] == 0:
    #             best['dist'] = dist
    #         elif dist < best['dist']:
    #             best['dist'], best['nick_name'], best['frame'] = dist, face_db['nick_name'], face_db['frame']
    # try:
    #     best_frame = best['frame'].astype(np.uint8)
    #     compare = cv.hconcat([cur_align, best_frame])
    #     print(f"비교 결과 : {best['nick_name']}")
    #     cv.imshow('Current & Detect result', compare)
    #     cv.waitKey(0)
    # except KeyError as e:
    #     print("현재 얼굴과 유사한 얼굴 db가 없습니다 !!!")
    #     return 0


db = load_database(db_path)
print(len(db))

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
            # 검출된 얼굴이 1개이고 정확도가 99% 이상인 경우
            if len(faces) is 1 and faces[0]['confidence'] >= 0.99:
                temp_conf_list.append(faces[0]['confidence'])
                temp_frame_list.append(frame)
                temp_fps_list.append(current_fps)
                frame = cv.rectangle(frame, box_range[0], box_range[1], red_color, box_thickness)

                # 최근 3초 동안 얼굴 인식 정확도가 0.999 이상이면
                if len(temp_conf_list) >= int(3 * (sum(temp_fps_list) / len(temp_fps_list))):
                    best_frame = choose_best_frame(temp_conf_list, temp_frame_list)
                    best_face = detector.detect_faces(best_frame)
                    best_align = face_aligner.align(best_frame, best_face[0]['keypoints']['left_eye'],
                                                    best_face[0]['keypoints']['right_eye'])
                    best_align = cv.cvtColor(best_align, cv.COLOR_RGB2BGR)
                    best_embedded = l2_normalize(face_embedding(best_align)).tolist()
                    compare_face_db(db, best_embedded, best_align)
                    exit()
                    # cv.imshow('best_align', best_align)
                    # cv.waitKey(0)

            else:
                frame = cv.rectangle(frame, box_range[0], box_range[1], green_color, box_thickness)
                temp_conf_list.clear()
                temp_frame_list.clear()
                temp_fps_list.clear()
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
