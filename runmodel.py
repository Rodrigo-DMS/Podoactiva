import time

startTime = time.time()
import matplotlib

matplotlib.use('TkAgg')  # TkAgg
import matplotlib.pyplot as plt
import os
import cv2
import subprocess
import numpy as np
from sklearn.decomposition import PCA
import math
import imageio
from concurrent.futures import ProcessPoolExecutor
from typing import List


# FUNCTIONS

# CONVERT VIDEO TO IMAGES
# 1
def crop_black_bars(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold the frame
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding box of the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_frame = frame[y:y + h, x:x + w]
        return cropped_frame
    else:
        return frame


def process_frame(frame):
    cropped_image = crop_black_bars(frame)
    return cropped_image


def save_frames(input_video_path, output_dir, frame_interval=1, batch_size=100):
    print("Converting Video to Images...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    reader = imageio.get_reader(input_video_path)
    total_frames = len(reader)

    saved_count = 0
    frame_batch = []

    with ProcessPoolExecutor() as executor:
        for i, frame in enumerate(reader):
            if i % frame_interval == 0:
                frame_batch.append(frame)

                # If batch is full or it's the last batch
                if len(frame_batch) == batch_size or i >= total_frames - frame_interval:
                    results = list(executor.map(process_frame, frame_batch))

                    for result in results:
                        output_path = os.path.join(output_dir, f"frame{saved_count}.jpg")
                        imageio.imwrite(output_path, result)
                        saved_count += 1
                    frame_batch = []

    print(f'Read {total_frames} frames, saved {saved_count} frames in {output_dir}')


# POSTPROCESSING FUNCTIONS
def select_white_region(mask):
    """
  From a mask, returns the middle point of the leg and the width of the leg
  at each row.
  """
    width = mask.shape[1]  # y size of image
    mid_points = []
    leg_width = []
    for i, row in enumerate(mask):  # select only the points where the mask is white
        first_pos = np.argmax(row)  # saves the index of the first white point in the row
        last_pos = width - np.argmax(
            row[::-1]) - 1  # flips the list with [::-1] and saves the index of the first white point (last)

        if first_pos != 0:
            mid_points.append(
                [int((last_pos - first_pos) // 2) + first_pos, i])  # get middle point of the mask at the given y
            leg_width.append((last_pos - first_pos))  # get distance between first and last point

    return mid_points, leg_width


def get_ankle_knee_pos(leg_widths, top_cut=0.1, knee_cut=0.3, ankle_cut=0.3):
    """
  Splits the mask of the leg in two: (1) knee to ankle and (2) ankle to ground by computing the two narrowest points in the leg
  """
    # define the limits in which to search for the minimums

    total_height_leg = len(leg_widths)  # height of the mask
    min_knee_y = int(
        total_height_leg * top_cut)  # from which point to start looking for the minimum (top part of the leg)
    max_knee_y = int(total_height_leg * knee_cut)  # until which point to look for the minimum (top part of the leg)
    min_ankle_y = int(total_height_leg * (
            1 - ankle_cut))  # from which point to start looking for the minimum (bottom part of the leg)
    max_ankle_y = int(
        total_height_leg * (1 - top_cut))  # until which point to look for the minimum (bottom part of the leg)

    leg_widths = np.array(leg_widths)
    knee = np.argmin(leg_widths[min_knee_y:max_knee_y])
    ankle = np.argmin(leg_widths[min_ankle_y:max_ankle_y])

    upper_part_range = slice(knee + min_knee_y, ankle + min_ankle_y)
    lower_part_range = slice(ankle + min_ankle_y, None)

    return upper_part_range, lower_part_range


def compute_pca(points):
    """
  From a 2D array computes PCA to 1 dimensions and returns reprojected coordinates (line)
  """
    pca_h = PCA(1)  # reduce to 1 principal components.
    converted_data = pca_h.fit_transform(points)
    return pca_h.inverse_transform(converted_data)  # reproject into original coordinates


def compute_slope(points):
    """Compute the slope of the best-fit line for the given points."""
    x = points[:, 0]
    y = points[:, 1]

    A = np.vstack([x, np.ones(len(x))]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]

    return m


def angle_with_vertical(slope):
    """Compute the angle between a line with given slope and a vertical line."""
    if math.isinf(slope):  # If the line is vertical
        return 0
    alpha = math.atan(slope)
    angle = math.degrees(math.pi / 2 - alpha)

    if angle < 15:  ###  HARD CODE FIX, SOME ANGLES WILL BE INACCURATE
        angle = abs(angle - 180)

    return angle


def compute_lines_angle(mask_dict):
    """
  Given a mask dict containig the category and the mask,
  returns the lines between the floor and the ankle, the ankle and the knee
  and the exterior angle they form.
  """
    category = mask_dict['cat']
    mask = np.array(mask_dict['mask'])

    mid_points, leg_widths = select_white_region(mask)  # get the white region
    upper_part_range, lower_part_range = get_ankle_knee_pos(leg_widths)  # get the two regions (up and down)

    mid_points_upper = np.array(mid_points[upper_part_range])  # select the mid points
    mid_points_lower = np.array(mid_points[lower_part_range])

    pca_upper = compute_pca(mid_points_upper)  # PCA

    point_max_y = [i for i in mid_points_lower[np.argmax(mid_points_lower[:, 1])]]  # point with maximum y

    min_y = np.min(mid_points_lower[:, 1])  # Minimum y

    point_min_y = point_max_y[0], min_y

    # slopeUpper = compute_slope(mid_points_upper)

    slopeUpper = compute_slope(pca_upper)

    angle = angle_with_vertical(slopeUpper)

    straight_line_lower = np.vstack([point_max_y, point_min_y])

    return pca_upper, straight_line_lower, angle


def getLegData(predsFolder: str, K=5):
    left_leg_data = {}
    right_leg_data = {}

    X = []
    left_count = 0
    right_count = 0
    for path in os.listdir('./' + predsFolder):
        if path.endswith('npy'):
            preds = np.load(os.path.join('./' + predsFolder, path), allow_pickle=True).item()
            if preds['classes'].size != 0:
                if preds['classes'][0] == 0:
                    left_leg_data[left_count] = preds | {'img_path': path.split('.')[0]}
                    left_count += 1
                else:
                    right_leg_data[right_count] = preds | {'img_path': path.split('.')[0]}
                    right_count += 1

    # get the top K predictions for the left and right leg
    left_leg = dict(sorted(left_leg_data.items(), key=lambda x: x[1]['scores'][0], reverse=True)[0:K])
    right_leg = dict(sorted(right_leg_data.items(), key=lambda x: x[1]['scores'][0], reverse=True)[0:K])

    left_leg = {k: {'mask': v['masks'][0], 'cat': 1, 'img_path': v['img_path']} for k, v in left_leg.items()}
    right_leg = {k: {'mask': v['masks'][0], 'cat': 2, 'img_path': v['img_path']} for k, v in right_leg.items()}
    all_legs = list(left_leg.values()) + list(right_leg.values())

    for leg in all_legs:  # compute lines and angle for left leg
        # print(leg["img_path"])  ## CHECK IF FRAMES ARE NOT ADJACENT
        high, low, angle = compute_lines_angle(leg)
        leg.update({'highs': high, 'lows': low, 'angle': angle})

    return all_legs


# PLOT FUNCTIONS
def find_intersection(pca_upper, vertical_line):
    # Find slope of pca_upper
    x1, y1, x2, y2 = pca_upper[0][0], pca_upper[0][1], pca_upper[1][0], pca_upper[1][1]
    m = (y2 - y1) / (x2 - x1)

    # x-coordinate of vertical line
    x_vertical = vertical_line[0][0]

    # Using the equation of the line to find y-coordinate of intersection point
    y_intersection = m * (x_vertical - x1) + y1

    return x_vertical, y_intersection


def display_predictions_v7(legs, num_display=5):
    """
    Display mask predictions for both left and right legs in the same figure.

    Args:
    - legs (list): List containing dictionaries with mask predictions and corresponding image data.
    - num_display (int): Number of images to display for each leg type. Default is 5.
    """

    mask_color = (200, 200, 200)  # Light gray color

    # Filter by cat
    left_legs = [leg for leg in legs if leg['cat'] == 1][:num_display]
    right_legs = [leg for leg in legs if leg['cat'] == 2][:num_display]

    # Create a 2x5 figure with subplots
    fig, axes = plt.subplots(2, num_display, figsize=(15, 10))

    # Helper function to display leg predictions on axes
    def display_on_axes(ax, leg, localReachVertical=REACH_VERTICAL):
        # Original image
        original_image = cv2.imread(PATH + "/" + IMAGES_FOLDER + "/" + leg['img_path'] + ".jpg")
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB format

        # Color mask
        mask = leg['mask']
        colored_mask = (mask[..., None] * mask_color).astype(original_image.dtype)

        # Plot 30% transparency mask over original image
        overlayed_image = cv2.addWeighted(original_image, 1.0, colored_mask, 0.3, 0)

        # Compute the lines and angle
        # pca_upper, line_lower, angle = compute_lines_angle(leg)
        pca_upper, line_lower, angle = leg['highs'], leg['lows'], leg['angle']

        if localReachVertical:
            intersection_point = find_intersection(pca_upper, line_lower)  ## DRAW UPPER UNTIL VERTICAL
            # Plot the lines on top of the mask
            ax.plot([pca_upper[0][0], intersection_point[0]], [pca_upper[0][1], intersection_point[1]],
                    'r-')  ## SET VERTICAL AS ENDPOINT
        else:
            # Plot the lines on top of the mask
            ax.plot(pca_upper[:, 0], pca_upper[:, 1], 'r-')  # Plotting pca_upper in red

        ax.plot(line_lower[:, 0], line_lower[:, 1], 'b-')  # Plotting pca_lower in blue

        # Display the image
        ax.imshow(overlayed_image)
        ax.axis('off')

        # Display angle below the image, color-coded based on its value
        angle_color = 'red' if angle > 175 else 'green'
        ax.set_title(f"Angle: {angle:.2f}", color=angle_color)

    def checkAngle(ang):

        if ang < 170:
            color = 'red'  # Overpronator
            classification = "Overpronator"
        elif ang <= 174:
            color = 'green'  # Neutral
            classification = "Neutral"
        else:
            color = 'orange'  # Supinator
            classification = "Supinator"

        return classification, color

    leftAngles = []
    rightAngles = []
    for i, leg in enumerate(left_legs[:5]):
        display_on_axes(axes[0, i], leg)
        angle = leg.get('angle', None)
        if angle is not None:
            leftAngles.append(angle)
            pronType, color = checkAngle(angle)
            axes[0, i].set_title(f"{pronType} ({angle:.2f}) ({leg['img_path'].split('e')[1]})", color=color)

    for i, leg in enumerate(right_legs[:5]):
        display_on_axes(axes[1, i], leg)
        angle = leg.get('angle', None)
        if angle is not None:
            rightAngles.append(angle)
            pronType, color = checkAngle(angle)
            axes[1, i].set_title(f"{pronType} ({angle:.2f}) ({leg['img_path'].split('e')[1]})", color=color)

    # Add titles to the rows
    # fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.1)
    fig.suptitle("    Results", ha='center', va='center', fontsize=18)

    meanLefts = np.round(np.mean(leftAngles), 2)
    meanRights = np.round(np.mean(rightAngles), 2)

    # Here's how you add the row titles:
    fig.text(0.5, 1.15, 'Left', ha='center', va='center', transform=axes[0, 2].transAxes, fontsize=14)
    fig.text(0.5, 1.15, 'Right', ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=14)

    fig.text(0.15, 0.07, f'LEFT LEG:          MEAN ANGLE: {meanLefts}      PRONATION: {checkAngle(meanLefts)[0]}')
    fig.text(0.15, 0.03, f'RIGHT LEG:        MEAN ANGLE: {meanRights}      PRONATION: {checkAngle(meanRights)[0]}')
    plt.show()


# MAIN
def main():
    # PATH AND DIRECTORY
    # Declare the path to the folder where you have YOLACT installed
    global PATH, IMAGES_FOLDER, REACH_VERTICAL
    PATH = r'C:\Users\User\PycharmProjects\gitPodoactiva\YOLACT_'  # Adjust this path to where you've saved YOLACT locally  ## INPUT REQUIRED
    os.chdir(PATH)  # Change directory
    VIDEO_PATH = 'F:/00-podoactiva/Videos/sample_20.MOV'  ## INPUT REQUIRED
    ID = "123"
    print(f"PATH: {VIDEO_PATH}    ID: {ID}")

    start_read = time.time()

    # Replace with your local video path

    # Create directory for converted images
    IMAGES_FOLDER = ID + "_IMAGES"
    os.makedirs(IMAGES_FOLDER, exist_ok=True)

    # Convert frames
    save_frames(VIDEO_PATH, './' + IMAGES_FOLDER, frame_interval=3)  # Set desired frame interval

    # Create directory for outputted predictions
    PREDICTIONS_FOLDER = ID + '_PREDICTIONS'
    os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)
    read_time = time.time() - start_read
    print(f'Read Time: {read_time}')

    def check():
        # Checks
        import torchvision, torch
        print(torchvision.__version__)
        print(torch.__version__)
        print("GPU:", torch.cuda.is_available())
        return None

    # check()
    # RUN YOLACT MODEL
    print('Running model...')
    model_start_time = time.time()
    subprocess.run(["python", "eval.py",
                    "--cuda", "True",  ## USE GPU
                    "--trained_model", "interrupt",  ## WEIGHTS TO USE
                    "--config", "yolact_resnet50_foot_pron_config",
                    "--score_threshold", "0.2",  ## MINIMUM PREDICTION CONFIDENCE
                    "--top_k", "1",  ## MAXIMUM PREDICTIONS
                    "--output_coco_json",
                    "--images", f"./{IMAGES_FOLDER}/|./{PREDICTIONS_FOLDER}/"])  ## INPUT FOLDER:OUTPUT FOLDER
    print(f'Read Time: {read_time}')
    print("Model Time:", time.time() - model_start_time)
    # Preprocessing
    start_postprocessing = time.time()
    all_legs = getLegData(PREDICTIONS_FOLDER)
    print(f"Time Post-processing: {time.time() - start_postprocessing}")
    # Display
    start_display = time.time()
    REACH_VERTICAL = False
    display_predictions_v7(all_legs, 5)
    print(f"REACH_VERTICAL = {REACH_VERTICAL}")
    print(f"Time Display: {time.time() - start_display}")

    end_time = time.time()
    print(f"Time: {end_time - startTime}")


if __name__ == '__main__':
    # from multiprocessing import freeze_support
    # freeze_support()  # If you're planning to turn this into an executable
    main()
