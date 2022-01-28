import cv2 as cv
import numpy as np
import json
import time
import os
import random


def same_face(embedding_path, save_path, same_count):
    files = os.listdir(embedding_path)
    for i in range(0, len(files)-1, 2):
        tmp1 = np.load(embedding_path + '/' + files[i])
        tmp2 = np.load(embedding_path + '/' + files[i + 1])
        # print(f'tmp1: {tmp1}')
        # print(f'tmp2: {tmp2}')
        # print(f'sum: {tmp1 + tmp2}')
        # print(f'test: {(tmp1 + tmp2)**2}')
        # print(f'test: {(tmp1 - tmp2) ** 2}')
        result = ((tmp1 - tmp2) ** 2)
        # np.save(save_path + '/' + str(same_count) + '.npy', (tmp1 - tmp2)**2)
        np.save(save_path + '/' + str(same_count) + '.npy', result)
        same_count += 1
    return same_count


def different_face(embedding_path1, embedding_path2, save_path, differ_count):
    path1_files = os.listdir(embedding_path1)
    path2_files = os.listdir(embedding_path2)
    for path1_file in path1_files:
        tmp1 = np.load(embedding_path1 + '/' + path1_file)
        tmp2 = np.load(embedding_path2 + '/' + random.sample(path2_files, 1)[0])
        # np.save(save_path + '/' + str(differ_count) + '.npy', np.hstack([tmp1, tmp2]))
        result = ((tmp1 - tmp2) ** 2)
        # np.save(save_path + '/' + str(differ_count) + '.npy', (tmp1 - tmp2) ** 2)
        np.save(save_path + '/' + str(differ_count) + '.npy', result)
        differ_count += 1
    return differ_count


save_path = 'C:/Users/newms_bxz/Desktop/final senial project/vggface2_train/preprocessed data 3'
# save_path = 'C:/Users/newms_bxz/Desktop/final senial project/vggface2_test/preprocessed data 2'
embedding_path = 'C:/Users/newms_bxz/Desktop/final senial project/vggface2_train/embedding'
# embedding_path = 'C:/Users/newms_bxz/Desktop/final senial project/vggface2_test/test_embedding(all data) 2'

humans = os.listdir(embedding_path)
human_count = len(humans)
print(human_count)

count, label = 0, 1
same_count, differ_count = 0, 0

while True:
    if label == 1:
        same_count = same_face(embedding_path + '/' + humans[count], save_path + '/same', same_count)
        print(humans[count])
        count += 1
        if count >= human_count - 1:
            break
        label = 0
    elif label == 0:
        differ_count = different_face(embedding_path + '/' + humans[count], embedding_path + '/' + humans[count + 1],
                                      save_path + '/different', differ_count)
        print(humans[count])
        count += 2
        if count >= human_count - 1:
            break
        label = 1
