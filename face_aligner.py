from typing import Tuple

import cv2 as cv
import numpy as np


class FaceAligner:
    def __init__(self, desired_left_eye=(0.3, 0.3), desired_face_width=160, desired_face_height=160):
        self.desired_left_eye = desired_left_eye
        # compute the desired right eye x-coordinate based on the desired x-coordinate of the left eye
        self.desired_right_eye = (1 - desired_left_eye[0], desired_left_eye[1])
        self.desired_iod = self.desired_right_eye[0] - self.desired_left_eye[0]
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height
        self.desired_dist = self.desired_iod * self.desired_face_width
        self.tx = self.desired_face_width // 2
        self.ty = self.desired_face_height * self.desired_left_eye[1]

    def align(self, img: np.ndarray, left_eye: Tuple[int, int], right_eye: Tuple[int, int]):
        # compute the angle between two eyes
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.linalg.norm([dx, dy])
        scale = self.desired_dist / dist

        # compute center (x, y) coordinates between the two eyes in the input image
        eyes_center = (left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2

        # grab the rotation matrix for rotating and scale the face
        matrix = cv.getRotationMatrix2D(eyes_center, angle, scale)

        # update the translation component of the matrix
        matrix[0, 2] += self.tx - eyes_center[0]
        matrix[1, 2] += self.ty - eyes_center[1]

        # apply the affine transformation
        output = cv.warpAffine(img, matrix, (self.desired_face_width, self.desired_face_height), flags=cv.INTER_CUBIC)

        return output
