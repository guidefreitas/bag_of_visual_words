import matplotlib.pyplot as plt  # plt.plot(x,y)  plt.show()
import cv2
import time



def test_feature_detector(detector, imfname):
    image = cv2.imread(imfname)
    forb = cv2.FeatureDetector_create(detector)
    # Detect crashes program if image is not greyscale
    t1 = time.time()
    kpts = forb.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    t2 = time.time()
    print detector, 'number of KeyPoint objects', len(kpts), '(time', t2-t1, ')'

    return kpts


def main():
    imfname = r'/Volumes/Imagens/GUI003/train/guitarra/51gF65a7VuL._AA160_.jpg'


    detector_format = ["","Grid","Pyramid"]
    # "Dense" and "SimpleBlob" omitted because they caused the program to crash
    detector_types = ["FAST","STAR","SIFT","SURF","ORB","MSER","GFTT","HARRIS"]


    for form in detector_format:
        for detector in detector_types:
            kpts = test_feature_detector(form + detector, imfname)

            # KeyPoint class: angle, class_id, octave, pt, response, size
            plt.figure(form + detector)
            for k in kpts:
                x,y = k.pt
                plt.plot(x,-y,'ro')
            plt.axis('equal')

    plt.show()


if __name__ == '__main__':
    main()