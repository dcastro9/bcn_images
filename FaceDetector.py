import cv2
from os import listdir
from os.path import isfile, join

import csv

class FaceDetector(object):
    """ Processes an image and returns the cropped face for a given length
    and width. Returned image will be cropped to that size.

    Attributes:
       dimension: Tuple of length 2, contains the length & width of desired
                  face size.
    """

    def __init__(self, dimension,
                 cascade_fn="cascades/haarcascade_frontalface_alt.xml",
                 nested_fn="cascades/haarcascade_eye.xml"):
        """Creates a class to detect faces in an image.
        """
        self._dimension = dimension
        self._cascade = cv2.CascadeClassifier(cascade_fn)
        self._nested = cv2.CascadeClassifier(nested_fn)

    def __detect(self, img):
        """Private method to process the classifiers on an image.
        """
        rects = self._cascade.detectMultiScale(img,
                                               scaleFactor=1.5,
                                               minNeighbors=1,
                                               minSize=(50, 50),
                                               flags = cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        rects[:,2:] += rects[:,:2]
        return rects

    def process(self, img):
        """Processes any given image and returns the largest face it finds.

        Returns:
           A cropped image of the face.
        """

        rects = self.__detect(img)
        if len(rects) == 0:
          return None

        max_idx = 0
        max_area = 0
        cur_idx = 0
        for x1, y1, x2, y2 in rects:
          if (y2 - y1)*(x2 - x1) > max_area:
            max_idx = cur_idx
            max_area = (y2 - y1)*(x2 - x1)
          cur_idx += 1

        big_face_rect = rects[max_idx]

        if max_area < 1000:
          return None

        for x1, y1, x2, y2 in [big_face_rect]:
            height = float(round(y2-y1))
            width = float(round(x2-x1))
            dim_y = height / self._dimension[0]
            dim_x = width / self._dimension[1]

            if dim_y > 0 and dim_x > 0:
                y_start = max(y1 - 550, 0)
                y_end = y2 + 550
                y_slice = max(int(round((y2 - y1) * 9 / 
                                        (self._dimension[0] * 11))), 1)
                
                x_start = max(x1 - 450, 0)
                x_end = x2 + 450
                x_slice = max(int(round((x2 - x1) /
                                        self._dimension[1])), 1)
                # print ("Y(%.2f, %.2f, %.2f)" % (y_start, y_end, y_slice))
                # print ("X(%.2f, %.2f, %.2f)" % (x_start, x_end, x_slice))
                img = img[y_start: y_end: y_slice,
                          x_start: x_end: x_slice]
                
                if img.shape[0] != 0:
                  return img
        return None

# CSV Reader
mapping = {}
with open("names_students.csv", "rb") as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  for row in reader:
    mapping[row[0].lower()] = row[1]

face_detector = FaceDetector((360,440))
img_path = "faces/"
out_path = "out_faces/"

# For images in folder.
images = [f for f in sorted(listdir(img_path)) if isfile(join(img_path,f))]
for image in images:
  name = image[:-4]
  my_image = cv2.imread(img_path + image)
  out = face_detector.process(my_image)
  # cv2.imwrite(join(out_path, "MNTPERSONAtemp-" + name.lower() + ".jpg"),
  #                 out)

  if out.any() != None:
    height, width, dim = out.shape
    # This is rather confusing.
    desired_height = width*11/9 - (width*11/9)%11
    desired_width = height*9/11 - (height*9/11)%9
    crop_w_amount = (width-desired_width)/2
    if (desired_height < height):
      out = out[:desired_height,:desired_height*9/11]
    else:
      out = out[:desired_width*11/9,crop_w_amount:width-crop_w_amount]
    if name.lower() in mapping.keys() and out.shape[0] > 0:
      out = cv2.resize(out, (0,0), fx=440./(out.shape[0]), fy=440./(out.shape[0]))
      cv2.imwrite(join(out_path, "MNTPERSONA-" + mapping[name.lower()] + ".jpg"),
                  out)
    else:
      print "Entry not found for " + name
      # cv2.imwrite(join(out_path, "MNTPERSONA-" + name.lower() + ".jpg"),
      #             out)
      cv2.imwrite(join(out_path, name.lower() + ".jpg"),
                  out)
  else:
    print "Face not found for " + name
  