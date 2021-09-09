import numpy as np
import cv2 

def convolve2d(image, kernel,size):
    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros_like(image)
    #adding pad= 2 to image
    pad = np.zeros((image.shape[0] + 4, image.shape[1] + 4))
    pad[2:-2, 2:-2] = image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            output[y, x]=(kernel * pad[y: y+size, x: x+size]).sum()
    return output

def gradient_intensity(img):
    #sobel filters
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],np.int32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],np.int32)
    #apply sobels
    Ix = convolve2d(img, Kx,3)
    Iy = convolve2d(img, Ky,3)

    G = np.hypot(Ix, Iy)
    D = np.arctan2(Iy, Ix)
    #return Ix ,not G since we need the vertical edges
    return (Ix,G,D)
def round_angle(angle):
    angle=np.rad2deg(angle)%180
    if(0 <=angle<22.5 or 157.5<=angle<180):
        angle=0
    elif 22.5<=angle<67.5:
        angle=45
    elif 67.5<=angle<112.5:
        angle=90
    elif 112.5<=angle<157.5:
        angle=135
    return angle

def suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    for i in range(M):
        for j in range(N):
            # find neighbour pixels to visit from the gradient directions
            where = round_angle(D[i, j])
            try:
                if where == 0:
                    if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                        Z[i, j] = img[i, j]
                elif where == 90:
                    if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
                        Z[i, j] = img[i, j]
                elif where == 135:
                    if (img[i, j] >= img[i - 1, j - 1]) and (img[i, j] >= img[i + 1, j + 1]):
                        Z[i, j] = img[i, j]
                elif where == 45:
                    if (img[i, j] >= img[i - 1, j + 1]) and (img[i, j] >= img[i + 1, j - 1]):
                        Z[i, j] = img[i, j]
            except IndexError as e:
                pass
    return Z


def threshold(img, t, T):
    # define gray value of a WEAK and a STRONG pixel
    cf = {
        'WEAK': np.int32(10),
        'STRONG': np.int32(255),
    }
    # get strong pixel indices
    strong_i, strong_j = np.where(img > T)
    # get weak pixel indices
    weak_i, weak_j = np.where((img >= t) & (img <= T))
    # get pixel indices set to be zero
    zero_i, zero_j = np.where(img < t)
    # set values
    img[strong_i, strong_j] = cf.get('STRONG')
    img[weak_i, weak_j] = cf.get('WEAK')
    img[zero_i, zero_j] = np.int32(0)

    return (img, cf.get('WEAK'))

def tracking(img, weak, strong=255):
    M, N = img.shape
    for i in range(M):
        for j in range(N):
            if img[i, j] == weak:
                # check if one of the neighbours is strong (=255 by default)
                try:
                    if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong)
                        or (img[i, j + 1] == strong) or (img[i, j - 1] == strong)
                        or (img[i + 1, j + 1] == strong) or (img[i - 1, j - 1] == strong)):
                            img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

if __name__ == "__main__":

    # Reading the image
    imgr = cv2.imread("sudoku.jpg")
    # read in gray
    img = cv2.imread("sudoku.jpg",0)
    img = np.asarray( img, dtype="int32" )
    # gaussian filter 5x5
    kernel = np.array([[1,4,6,4, 1], [4, 16, 24,16,4], [6, 24, 36,24,6],[4,16,24,16,4],[1,4,6,4,1]])/256
    # applying gaussian filter number of times to smooth the image and get real edges
    smoothy= convolve2d(img,kernel,5)
    smoothy= convolve2d(smoothy,kernel,5)
    smoothy= convolve2d(smoothy,kernel,5)
    smoothy= convolve2d(smoothy,kernel,5)
    # sobel filter
    sobely,G,D=gradient_intensity(smoothy)
    # supression 
    supressed = suppression(np.copy(sobely), D)
    # threshold
    thresh, weak= threshold(np.copy(supressed), 50 , 60 )
    # tracking
    tracked = tracking(np.copy(thresh), weak)
    # get the strong edges of the tracked image
    strong_i, strong_j = np.where(tracked ==255)
    # convert to a list to use some list functions
    strong_j=(list(strong_j))
    # just taking fifth of the list(points) is enough to detect the lines 
    strong_j=strong_j[:len(strong_j)//5]
    # loop over strong J points and check if theres more than 24 points(considering 24 is the minimum of a line length) on the same x axis (get only vertical lines ) and draw a line there 
    for x in strong_j:
        if(strong_j.count(x)+strong_j.count(x-1)>24):
            # if a line was found draw a line from that point to the end of the image 
            applyLines=cv2.line(imgr, (x, 0), (x, imgr.shape[1]), (0, 0, 255), 1)
    cv2.imshow('image',applyLines)
    cv2.waitKey()   
