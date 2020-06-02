import cv2
import os
import re

def make_movie(image_folder, video_name='hep_video.avi'):

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
