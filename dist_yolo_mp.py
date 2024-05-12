import json
import cv2
from PIL import Image
import numpy as np
import json
from os import remove
import subprocess as sp
import time
import multiprocessing as mp
import sys

# Process video image detection
def process_video_multiprocessing(group_number, frame_jump_unit, file_name):
    sys.path.append('C:/Bryan/TA/Integrated')
    from dist_yolo.dist_yolo import DistYOLO
    # Load DistYOLO
    dist_yolo = DistYOLO()
    dist_yolo.load_model('./dist_yolo/trained_final (1).h5')

    # Read video file
    cap = cv2.VideoCapture(file_name)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_jump_unit * group_number)

    # get height, width and frame count of the video
    width, height = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
    no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    proc_frames = 0

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # out = cv2.VideoWriter()
    # output_file_name = "output_multi.mp4"
    # out.open("output_{}.mp4".format(group_number), fourcc, fps, (width, height), True)

    frames_detections = []
    try:
        while proc_frames < frame_jump_unit:
            ret, frame = cap.read()
            if not ret:
                break

            im = frame
            # Perform face detection on each frame
            # _, bboxes = detectum.process_frame(im, THRESHOLD)

            # Perform Dist-YOLO detection on each frame
            result = dist_yolo.detect(Image.fromarray(im))
            result_zips = list(zip(result[0].tolist(), result[1].tolist(), result[2].tolist(), result[3].tolist()))
            dets = []
            for result_zip in result_zips:
                temp = []
                for c in result_zip:
                    if hasattr(c, "__len__"):
                        for cc in c:
                            temp.append(cc)
                    else:
                        temp.append(c)
                dets.append(temp)

            # Add to frames detections
            frames_detections.append(dets)
            # frames_detections.append(frame_jump_unit * group_number + proc_frames)

            # Loop through list (if empty this will be skipped) and overlay green bboxes
            # for i in bboxes:
            #     cv2.rectangle(im, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 3)
            
            # write the frame
            # out.write(im)

            proc_frames += 1
    except:
        pass
        # Release resources
        cap.release()
        # out.release()

    # Release resources
    cap.release()
    # out.release()
    with open(f'output/dist_yolo_{group_number}.json', 'w', encoding='utf-8') as f:
        json.dump({ "frames_detections": frames_detections }, f, ensure_ascii=False, indent=4)
    # out.open("output_{}.mp4".format(group_number), fourcc, fps, (width, height), True)

def combine_output_files(num_processes):
    # Create a list of output files and store the file names in a txt file
    list_of_output_files = [f"output/dist_yolo_{i}.json".format(i) for i in range(num_processes)]
    final_frames_detections = []
    for t in list_of_output_files:
        with open(t) as f:
            frames_detections = json.load(f)
            final_frames_detections += frames_detections['frames_detections']
    with open(f'output/dist_yolo_output.json', 'w', encoding='utf-8') as f:
        json.dump({ "final_frames_detections": final_frames_detections }, f, ensure_ascii=False, indent=4)
            

    # use ffmpeg to combine the video output files
    # ffmpeg_cmd = "ffmpeg -y -loglevel error -f concat -safe 0 -i list_of_output_files.txt -vcodec copy " + output_file_name
    # sp.Popen(ffmpeg_cmd, shell=True).wait()

    # Remove the temperory output files
    for f in list_of_output_files:
        remove(f)
    # remove("list_of_output_files.txt")

    # Create a list of output files and store the file names in a txt file
    # list_of_output_files = ["output_{}.mp4".format(i) for i in range(num_processes)]
    # with open("list_of_output_files.txt", "w") as f:
    #     for t in list_of_output_files:
    #         f.write("file {} \n".format(t))

    # # use ffmpeg to combine the video output files
    # ffmpeg_cmd = "ffmpeg -y -loglevel error -f concat -safe 0 -i list_of_output_files.txt -vcodec copy " + output_file_name
    # sp.Popen(ffmpeg_cmd, shell=True).wait()

    # # Remove the temperory output files
    # for f in list_of_output_files:
    #     remove(f)
    # remove("list_of_output_files.txt")

def multi_process():
    print("Video processing using {} processes...".format(num_processes))
    start_time = time.time()

    # Paralle the execution of a function across multiple input values
    with mp.Pool(num_processes) as p:
        p.starmap(process_video_multiprocessing, [(i, frame_jump_unit, file_name) for i in range(num_processes)])

    print("Combining the processes...")
    combine_output_files(num_processes)

    end_time = time.time()

    total_processing_time = end_time - start_time
    print("Time taken: {}".format(total_processing_time))
    print("FPS : {}".format(frame_count/total_processing_time))

if __name__ == '__main__':
    file_name = "video/video1.mp4"
    output_file_name = "output.mp4"
    # width, height, frame_count = get_video_frame_details(file_name)

    cap = cv2.VideoCapture(file_name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video frame count = {}".format(frame_count))
    cap.release()

    # print("Width = {}, Height = {}".format(width, height))
    num_processes = mp.cpu_count()
    print("Number of CPU: " + str(num_processes))
    frame_jump_unit =  frame_count// num_processes
    multi_process()

    # process_video_multiprocessing(2, frame_jump_unit, file_name)