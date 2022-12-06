import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from PIL import Image, UnidentifiedImageError


def blur_faces(
    img: np.ndarray, faces: list, kernal_size: tuple, rectangle: bool = False
):
    """
    This function takes in a list of faces (insightface.app.common.Face) and both draws the bounding boxes over the faces and blurs everything in that bounding box.
    :param img: the np.ndarray that insightface.data.get_image returns (param to get_image() is a path to an image)
    :param faces: a list of insightface.app.common.Face objects
    :param kernal_size: the size of the kernal that is used to blur the image, (70, 70) for example. (The bigger the kernal size, the more blur)
    :param rectangle: whether to show the rectangle around the faces or not (along with gender, age and facial landmarks like eyes, mouth and nose)
    :return: a np.ndarray that contains the original image with a bounding box around the faces as well as a blur inside of the bounding boxes
    """
    dimg = img.copy()
    for i in range(len(faces)):
        face = faces[i]
        box = face.bbox.astype(np.int)
        color = (0, 0, 255)
        # create the rectangle other face analytics
        if rectangle:
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            # dots on each important part of the face (eyes, mouth, nose)
            if face.kps is not None:
                kps = face.kps.astype(np.int)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)
            # gender and age
            if face.gender is not None and face.age is not None:
                cv2.putText(
                    dimg,
                    "%s,%d" % (face.sex, face.age),
                    (box[0] - 1, box[1] - 4),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 255, 0),
                    1,
                )

        try:
            # blur rectangles
            roi = dimg[box[1] : box[3], box[0] : box[2]]
            roi = cv2.blur(roi, kernal_size)
            dimg[box[1] : box[1] + roi.shape[0], box[0] : box[0] + roi.shape[1]] = roi
        except:
            # For images with a large amount of faces, sometimes some fail, not sure why yet but seems to have no effect on final result
            print("error in blur")
    return dimg


def get_face_data(image_path: str):
    """
    This function takes in a path to an image and outputs the img (np.ndarray) as well as the list of faces (insightface.app.common.Face).
    :param image_path: the string representing the path to the image
    :return: img (np.ndarray), faces (list)
    """
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))
    img = ins_get_image(image_path)
    faces = app.get(img)
    return img, faces


def blur_directory():
    sourcedir = args.sourcedir
    outputdir = args.outputdir
    path = Path(sourcedir).absolute()
    for root, subdir, files in os.walk(sourcedir):
        ind = root.find(str(path.name))
        short_path = str(root)[ind:]
        new_path = outputdir + short_path
        try:
            os.mkdir(new_path)
        except OSError as error:
            print(error)

        for file in files:
            try:
                file_path = Path(root).absolute() / file
                # test if file is an image
                img = Image.open(str(file_path.absolute()))
                # format = img.format
                # img, faces = get_face_data(os.path.splitext(path)[0])
                # get rid of extension
                img, faces = get_face_data(os.path.splitext(str(file_path))[0])
                dimg = blur_faces(img, faces, (70, 70))
                output_path = Path(new_path).absolute() / file
                cv2.imwrite(str(output_path), dimg)
            except UnidentifiedImageError as error:
                print(error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    This is the detect faces sample using python language
    """
    )
    parser.add_argument(
        "--sourcedir",
        required=True,
        help="Path to the directory with the images with face data to detect",
    )
    parser.add_argument(
        "--outputdir",
        required=True,
        help="Path to the directory where the blurred images should be stored",
    )
    args = parser.parse_args()

    # example usage since this file is meant to be imported
    blur_directory()
