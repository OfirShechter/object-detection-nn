#%%
import sys
import os

# Add the root directory of your project to the PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(f"Adding {project_root} to PYTHONPATH")
sys.path.append(project_root)

import numpy as np

from nn.single_class_single_object_detection.object_class_detector import Object_Class_Detector

from nn.utils.apply_detector import apply_objects_detection
from nn.YOLO_VGG16.utils.helpers import convert_cells_to_bboxes, nms, plot_image
from nn.utils.frames_helpers import FrameHelpers
from nn.YOLO_VGG16.utils.constants import ANCHORS, image_size, device, s
import cv2
from PIL import Image
from nn.YOLO_VGG16.prepare_data.transforms import execute_transform
import torch
#%%
video_path = './video/dog_video_1.mp4'
frame_new_size = (image_size, image_size)
video, frames = FrameHelpers.get_video_and_frames(video_path, frame_new_size)

model_path = f"./saved_models/vgg_f_modele32_vgg16_checkpoint.pth.tar"
object_class_detector = Object_Class_Detector(model_path)

#%%
batch_size = 2
result_frames = []  # Buffer to store all processed frames
frame_index = 0 # Current frame index
while frame_index < len(frames):
    processed_frames = object_class_detector.plot_marked_images(
        frames[frame_index:frame_index+batch_size].copy())
    result_frames.extend(processed_frames)
    # print("index frame:", frame_index)
    frame_index += batch_size
#%%
# frames, processed_frames = apply_objects_detection(frames, object_class_detector, batch_size=2, show_frames=False)

#%%
# save results to video
output_path = 'result_videos/single_dog_detection.avi'
video_output_writer = FrameHelpers.get_video_writer(output_path, video, frame_new_size)
for frame in processed_frames:
    video_output_writer.write(frame.astype(np.uint8))

video.release()
video_output_writer.release()
# %%
