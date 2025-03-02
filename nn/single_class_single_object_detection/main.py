import time
import cv2
import numpy as np

from .object_class_detector import Object_Class_Detector

from ..utils.apply_detector import apply_objects_detection

from ..utils.frames_helpers import FrameHelpers
from nn.YOLO_VGG16.utils.constants import image_size

video_path = 'nn/single_class_single_object_detection/video/dog_video_1.mp4'
frame_new_size = (image_size, image_size)
video, frames = FrameHelpers.get_video_and_frames(video_path, frame_new_size)

object_class_detector = Object_Class_Detector()
frames, processed_frames = apply_objects_detection(frames, object_class_detector, show_frames=True)

# save results to video
output_path = 'result_videos/single_dog_detection.avi'
video_output_writer = FrameHelpers.get_video_writer(output_path, video, frame_new_size)
for frame in processed_frames:
    video_output_writer.write(frame.astype(np.uint8))

video.release()
video_output_writer.release()