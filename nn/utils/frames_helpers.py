from hmac import new
import cv2


class FrameHelpers:
    @staticmethod
    def get_video_and_frames(video_path):
        """
        Get all frames from a video.

        :param video: The video object
        :return: A list of all frames
        """
        video = cv2.VideoCapture(video_path)
        success, frame = video.read()
        frames = []
        success, frame = video.read()
        print("video was found:", success)
        counter = 100
        while success:
            frames.append(frame)
            success, frame = video.read()
            counter -= 1
            if counter == 0:
                break
        return video, frames

    @staticmethod
    def get_video_writer(output_video_path, video_capture, frame_size):
        """
        Get a video writer object.

        :param video_path: The path to the video
        :param frame_rate: The frame rate
        :param frame_size: The frame size
        :return: The video writer object
        """
        frame_rate = video_capture.get(
            cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        return cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)
