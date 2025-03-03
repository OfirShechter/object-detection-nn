#%%
import os

remote = True

if remote:
    base_path = '/home/dcor/niskhizov/Rar/object-detection-nn/nn'
    os.chdir('object-detection-nn')
else:
    base_path = 'nn'
    os.chdir('../..')

#%%
# !git pull

#%%
import numpy as np

from nn.single_class_single_object_detection.object_class_detector import Object_Class_Detector
from nn.utils.frames_helpers import FrameHelpers
from nn.YOLO_VGG16.utils.constants import image_size
import matplotlib.pyplot as plt
import cv2
#%%
video_path = f'{base_path}/single_class_single_object_detection/video/Best of 2024 Masters Agility Championships from Westminster Kennel Club _ FOX Sports.mp4'
frame_new_size = (image_size, image_size)
video, frames = FrameHelpers.get_video_and_frames(video_path)
#%%
model_path = f"{base_path}/YOLO_VGG16/degug_notebooks/vgg_f_modele32_vgg16_checkpoint.pth.tar"
# model_path = f"{base_path}/single_class_single_object_detection/saved_models/vgg_f_modele32_vgg16_checkpoint.pth.tar"
object_class_detector = Object_Class_Detector(model_path)
#%%
len(frames)
np.save(f'{base_path}/single_class_single_object_detection/frames.npy', frames)
#%%
batch_size = 16
result_frames = []  # Buffer to store all processed frames
frame_index = 0 # Current frame index
while frame_index < len(frames):
    processed_frames = object_class_detector.plot_marked_images(
        frames[frame_index:frame_index+batch_size].copy())
    result_frames.extend(processed_frames)
    print("index frame:", frame_index)
    frame_index += batch_size

#%%
# save result frames as pkl
np.save(f'{base_path}/single_class_single_object_detection/result_frames.npy', result_frames)

#%%
# load result frames from pkl
result_frames = np.load(f'{base_path}/single_class_single_object_detection/result_frames.npy', allow_pickle=True)
#%%
len(result_frames)
#%%
# write all frames to video
frame_new_size = (image_size, image_size)
output_path = f'{base_path}/dog_detection.avi'
frame_rate = 30
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(output_path, fourcc, frame_rate, frame_new_size)

for frame in result_frames:
    frame = (frame * 255).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    writer.write(frame)
writer.release()
# # %%
# np.array(result_frames).shape
# # %%
# import matplotlib.pyplot as plt
# plt.imshow(result_frames[15])
# plt.show()
# # %%
# np.max(result_frames[15].astype(np.uint8))
# # %%

# %%
