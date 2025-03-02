from multiprocessing import process
import cv2
import numpy as np

from .frames_helpers import FrameHelpers
import time

def apply_lane_detection(frames, object_class_detector, batch_size = 1, show_frames=False):
    result_frames = []  # Buffer to store all processed frames
    frame_index = 0 # Current frame index
    while frame_index < len(frames):
        processed_frames = object_class_detector.plot_marked_images(
            frames[frame_index:batch_size].copy())

        if show_frames:
            for i, processed_frame in enumerate(processed_frames):
                # Display the original frame
                displaying_frame = frames[frame_index + i].copy()
                cv2.imshow("Original", displaying_frame)
                cv2.imshow("Lane Detection", processed_frame)
                # time.sleep(0.1)

        # key = cv2.waitKey(0) & 0xFF
        # if key == ord('l'):  # 'l' for next frame
        #     frame_index = min(frame_index + 1, len(frames) - 1)
        #     continue
        # elif key == ord('k'):  # 'k' for previous frame
        #     frame_index = max(frame_index - 1, 0)
        #     continue
        # elif key == ord('q'):  # 'q' to quit
        #     break
        # Press Q on keyboard to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        result_frames.extend(processed_frames)
        # print("index frame:", frame_index)
        frame_index += batch_size

    # Release everything if job is finished
    cv2.destroyAllWindows()
    return (frames, processed_frames)