# Simple function used to process all the frames in a video clip
from moviepy.editor import VideoFileClip

def process_frame(image):
    return image

def process_clip(input_video_file, output_video_file, frame_processing_function):
    clip1 = VideoFileClip(input_video_file)
    white_clip = clip1.fl_image(frame_processing_function) #NOTE: this function expects color images!!
    white_clip.write_videofile(output_video_file, audio=False)

if __name__ == '__main__':
    import sys
    process_clip(sys.argv[1], sys.argv[2], process_frame)
