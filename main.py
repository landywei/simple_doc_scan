from skimage.filters import threshold_local
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
import os


def sort_four_points(points):
    mean_x = np.mean(points[:, 0])
    mean_y = np.mean(points[:, 1])
    sorted_ = np.zeros(points.shape)
    for i in points:
        if i[0] < mean_x and i[1] < mean_y:
            sorted_[0] = i
        elif i[0] > mean_x and i[1] < mean_y:
            sorted_[1] = i
        elif i[0] < mean_x and i[1] > mean_y:
            sorted_[2] = i
        elif i[0] > mean_x and i[1] > mean_y:
            sorted_[3] = i
        else:
            raise Exception("impossible point combos")
    return sorted_


def run_image(dir="images", image_name="demo.jpg", show=False, save=False):
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    image_n, ext = image_name.split(".")
    image = cv2.imread(os.path.join(dir, image_name))
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    h, w, _ = orig.shape
    image = imutils.resize(image, height=500)
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    """
    ====================
    edge detection stuff
    ====================
    """
    print("STEP 1: Edge Detection")
    edged = cv2.Canny(gray, 75, 200)
    if show:
        plt.figure()
        plt.imshow(edged, cmap="gray")
        # show the original image and the edge detected image
        plt.show()
    if save:
        cv2.imwrite(f"{dir}/{image_n}_edged.{ext}", edged)

    """
    ====================
    finding contours stuff
    ====================
    """
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    print("STEP 2: Find contours of paper")
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    else:
        print("no document found")
        return
    # show the contour (outline) of the piece of paper
    if show:
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
        plt.figure()
        plt.imshow(image)
        plt.show()
    if save:
        cv2.imwrite(f"{dir}/{image_n}_contour.{ext}", image)

    """
    ====================
    perspective transform stuff
    ====================
    """
    print("STEP 3: Apply perspective transform")
    # apply the four point transform to obtain a top-down
    # view of the original image

    # screenCnt: Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
    src = screenCnt.reshape(4, 2) * ratio
    src = sort_four_points(src)
    src = src.astype(np.float32)
    dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(orig, matrix, (w, h))
    resized = imutils.resize(warped, height=650)

    if show:
        # show the original and scanned images
        plt.figure()
        plt.imshow(resized)
        plt.show()
    if save:
        cv2.imwrite(f"{dir}/{image_n}_warped.{ext}", resized)

    """
    ====================
    binarization stuff
    ====================
    """
    # # convert the warped image to grayscale, then threshold it
    # # to give it that 'black and white' paper effect
    warped: np.ndarray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset=10, method="gaussian")
    warped = (warped > T).astype("uint8") * 255
    resized = imutils.resize(warped, height=650)
    if show:
        # show the original and scanned images
        plt.figure()
        plt.imshow(resized, cmap="gray")
        plt.show()
    if save:
        cv2.imwrite(f"{dir}/{image_n}_bin.{ext}", resized)


if __name__ == "__main__":
    for i in os.listdir("images"):
        if i[0] == "." or "_contour" in i or "_edge" in i or "_warped" in i or "_bin" in i:
            continue
        print("=========================", i)
        run_image(dir="images", image_name=i, show=False, save=True)
