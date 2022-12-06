import cv2
import numpy as np

def blend_with_mask_matrix(img, blurred, mask):
    res_channels = []
    for c in range(0, img.shape[2]):
        a = img[:, :, c]
        b = blurred[:, :, c]
        m = mask[:, :, c]
        res = cv2.add(
            cv2.multiply(b, cv2.divide(np.full_like(m, 255) - m, 255.0, dtype=cv2.CV_32F), dtype=cv2.CV_32F),
            cv2.multiply(a, cv2.divide(m, 255.0, dtype=cv2.CV_32F), dtype=cv2.CV_32F),
           dtype=cv2.CV_8U)
        res_channels += [res]
    res = cv2.merge(res_channels)
    return res


def blur_image(img, pts, kernel, blend_iterations):
    blurred = cv2.blur(img, kernel)
    mask = blurred.copy()
    fill_color = [255,255,255] 
    mask_value = 255
    pts = pts.reshape(-1, 1, 2) #flattens rows incase of weird points input
    stencil  = np.zeros(mask.shape[:-1]).astype(np.uint8)
    cv2.fillPoly(stencil, [pts], mask_value)
    sel = stencil != mask_value # select everything that is not mask_value
    mask[sel] = fill_color

    mask = cv2.blur(mask, kernel)
     
    res = blend_with_mask_matrix(img, blurred, mask)
    blend_iterations -= 1
    for i in range(blend_iterations):
        res = blend_with_mask_matrix(res, blurred, mask)
    return res


img = cv2.imread('/Users/ramsddc1/Documents/Projects/pii_gui/imgs/IMG_0220.JPG')
imgCopy = img.copy()
# the points will be taken from the pysimplegui (its width, height and not row, column)
pts = np.array([[0,0], [2000, 200], [3455, 5183]])
res = blur_image(imgCopy, pts, (71, 71), 4)
    
cv2.imwrite("/Users/ramsddc1/Documents/Projects/pii_gui/output/output.jpg",  res)
print("done")



