"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
import math

def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """
    x0=float(p0[0])
    y0=float(p0[1])
    x1=float(p1[0])
    y1=float(p1[1])
    return math.sqrt( (x0-x1)*(x0-x1) + (y0-y1)*(y0-y1) )


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    height=image.shape[0]
    width=image.shape[1]
    return [(0,0),(0,height-1),(width-1,0),(width-1,height-1)]


def sort_4(center): # center is np.array, [[x,y]]
    ans=[]
    #print center
    if center.shape[0]!=4:
        return None
    else:
        center = center.tolist()
    avex=0
    avey=0
    for i in range(4):
        avex+=center[i][0]
        avey+=center[i][1]
    ave=(avex/4,avey/4)
    #print ave
    #print center
    for tl in center:
        for bl in center:
            for tr in center:
                for br in center:
                    if (tl==bl or tl==tr or tl==br or bl==tr or bl==br or tr==br):
                        continue # non-identical
                    #print tl, bl, tr, br
                    tl_tl=tl[0]<ave[0] and tl[1]<ave[1]
                    tr_tr=tr[0]>ave[0] and tr[1]<ave[1]
                    bl_bl=bl[0]<ave[0] and bl[1]>ave[1]
                    br_br=br[0]>ave[0] and br[1]>ave[1]
                    if(tl_tl and tr_tr and bl_bl and br_br):
                        ans.append((tl[0],tl[1]))
                        ans.append((bl[0],bl[1]))
                        ans.append((tr[0],tr[1]))
                        ans.append((br[0],br[1]))
                        #print ans
                        return ans
                    longside=min(euclidean_distance(tl,tr), euclidean_distance(bl,br))
                    shortside=max(euclidean_distance(tl,bl),euclidean_distance(tr,br))
                    if (longside<=shortside):
                        continue #tl-tr and bl-br are long side, tl-bl, tr-br short side
                    widthvec=(tl[0],tl[1],bl[0],bl[1])
                    lengvec=(tl[0],tl[1],tr[0],tr[1])
                    diagvec=(tl[0],tl[1],br[0],br[1])
                    temp1=min(cos_ang(widthvec, diagvec), cos_ang(lengvec, diagvec))
                    temp2=cos_ang(widthvec,lengvec)
                    if temp2>temp1:#width and length forms 90 degree
                        continue

                   # print longside, shortside, tl, tr, bl, br
                    if (tl[0]<tr[0] and bl[0]<br[0] and tl[1]<bl[1] and tr[1]<br[1]):
                        ans.append((tl[0],tl[1]))
                        ans.append((bl[0],bl[1]))
                        ans.append((tr[0],tr[1]))
                        ans.append((br[0],br[1]))
                        #print ans
                        return ans
    return None

def cos_ang(i, j): # input i=(x1,y1,x2,y2)
    len1 = math.pow(i[2] - i[0], 2) + math.pow(i[3] - i[1], 2)
    len1 = math.sqrt(len1)
    len2 = math.pow(j[2] - j[0], 2) + math.pow(j[3] - j[1], 2)
    len2 = math.sqrt(len2)

    dot = (i[2] - i[0]) * (j[2] - j[0]) + (i[3] - i[1]) * (j[3] - j[1])
    return abs(dot / (len1 * len2))


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # find Harris corners
    gray = cv2.medianBlur(gray, 3)
    ksize = 15#15
    sigma = 35#6
    gray = cv2.GaussianBlur(gray, (ksize, ksize), sigma, sigma)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 7,3, 0.1)  # blocksize, sobel size, alpha
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.12 * dst.max(), 255, 0)  # value=max if > 0.01*max
    all_pts=np.transpose(np.nonzero(dst))
    all_pts[:, [0, 1]] = all_pts[:, [1, 0]]
    all_pts = np.array(all_pts)
    all_pts = np.float32(all_pts)  # kmeans can only pass float32
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1.0)
    ret, label, center = cv2.kmeans(all_pts, 4, criteria, 40,
                                    cv2.KMEANS_RANDOM_CENTERS)  # 4 is num of cluster, 10 is attempts

    if center is None:
        return None
    center = np.round(center)
    center = center.astype(int)
    ans=sort_4(center)
    #print ans
    return ans



def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    img=np.copy(image)
    cv2.line(img, markers[0], markers[1], (255, 0, 0), thickness)
    cv2.line(img, markers[0], markers[2], (255, 0, 0), thickness)
    cv2.line(img, markers[3], markers[1], (255, 0, 0), thickness)
    cv2.line(img, markers[3], markers[2], (255, 0, 0), thickness)
    return img


def project_imageA_onto_imageB(imageA, target_img, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    # """
    imageB=np.copy(target_img)
    inv_homo = np.linalg.inv(homography)
    inv_homo = inv_homo.astype('float32')
    map1 = np.fromfunction(lambda j, i: (inv_homo[0, 0] * i + inv_homo[0, 1] * j + inv_homo[0, 2]) / (
                inv_homo[2, 0] * i + inv_homo[2, 1] * j + inv_homo[2, 2]), (imageB.shape[0], imageB.shape[1]))
    map2 = np.fromfunction(lambda j, i: (inv_homo[1, 0] * i + inv_homo[1, 1] * j + inv_homo[1, 2]) / (
                inv_homo[2, 0] * i + inv_homo[2, 1] * j + inv_homo[2, 2]), (imageB.shape[0], imageB.shape[1]))
    map1 = map1.astype('float32')
    map2 = map2.astype('float32')
    imageB = cv2.remap(imageA, map1, map2, interpolation=cv2.INTER_LINEAR, dst=imageB,
                       borderMode=cv2.BORDER_TRANSPARENT)
    imageB=np.round(imageB)
    return imageB

def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    A = []
    for i in range(4):
        aui = dst_points[i][0]
        avi = dst_points[i][1]
        bui = src_points[i][0]
        bvi = src_points[i][1]
        A.append([bui, bvi, 1, 0, 0, 0, -bui * aui, -bvi * aui, -aui])
        A.append([0, 0, 0, bui, bvi, 1, -bui * avi, -bvi * avi, -avi])
    A = np.array(A)
    A=np.float32(A)
    #print A
    AT = np.transpose(A)
    M = np.dot(AT, A)
    val, vec = np.linalg.eig(M)
    index = 0
    mini = val[index]
    # print np.dot(M,(vec[:,index]))
    # print val[index]*vec[:,index]

    for i in range(vec.shape[0]):
        if (val[i] < mini):
            index = i
            mini = val[i]
    a = vec[:, index]
    # print a
    lst = []
    for b in range(3):
        lst.append([a[0 + 3 * b], a[1 + 3 * b], a[2 + 3 * b]])
    ans=np.array(lst)
    ans=ans/ans[2,2]
    return ans


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename) #return videocapture obj

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None
