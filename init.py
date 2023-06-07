import cv2
import numpy as np
import game_settings as settings

# import game settings
PERIOD = 10 * settings.INITIAL_PERIOD
SPEED = 10 * settings.INITIAL_SPEED
NUM_BALLS = settings.MAX_NUMBER_OF_BALLS
SPEED_INCREASE = settings.SPEED_INCREASE_PER_PERIOD
PERIOD_DECREASE = settings.PERIOD_DECREASE_PER_PERIOD
BALL_SIZE = settings.BALL_SIZE_IN_PIXELS
MARGIN = settings.LEFT_AND_RIGHT_MARGINS

# import cascade classifier which is used to detect faces
FACE_CASCADE = cv2.CascadeClassifier('frontal_face_cascade.xml')

# set up the camera video capture (capture, resolution, and frame rate)
RESOLUTIONS = {'1080p': (1920, 1080), '720p': (1280, 720), '480p': (640, 480)}
WIDTH = RESOLUTIONS[settings.CAMERA_RESOLUTION][0]
HEIGHT = RESOLUTIONS[settings.CAMERA_RESOLUTION][1]
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_XI_FRAMERATE, settings.CAMERA_FPS)

# set up game window size
WND_WIDTH = RESOLUTIONS[settings.SCREEN_RESOLUTION][0]
WND_HEIGHT = RESOLUTIONS[settings.SCREEN_RESOLUTION][1]
BLANK_FRAME = cv2.imread(f'images/{settings.THEME}_background.png')
BLANK_FRAME = cv2.resize(BLANK_FRAME, (WND_WIDTH, WND_HEIGHT))
DIFF_WIDTH = (WND_WIDTH - WIDTH) // 2
DIFF_HEIGHT = (WND_HEIGHT - HEIGHT) // 2

# read net image which is placed over the faces
NET_IMG = cv2.imread('images/net.png', cv2.IMREAD_UNCHANGED)

# set up the ball images which are the same image but each rotated in a certain angle
BALL_IMGS = []
for angle in range(18):
    img_read = cv2.imread(f'images/{angle * 20}.png', cv2.IMREAD_UNCHANGED)  # read image with its alpha channel
    img_resize = cv2.resize(img_read, (BALL_SIZE, BALL_SIZE))  # resize to specified ball size
    alpha_read = img_resize[:, :, 3] // 255  # get alpha channel of the image
    # concatenate alpha channel for further processing of ball image
    alpha_img = np.concatenate((alpha_read[:, :, None], alpha_read[:, :, None], alpha_read[:, :, None]), axis=2)
    img = img_resize[:, :, :3] * alpha_img  # take pixels only where alpha channel is 255 (max)
    BALL_IMGS.append((img, alpha_img))  # store ball image and its alpha channel as a tuple

# read pause image
PAUSE = cv2.imread('images/pause.png')
PAUSE = cv2.resize(PAUSE, (WND_WIDTH, WND_HEIGHT))

# calculate font scale and position of player score in the frame
FONT_SCALE = 3 * WND_HEIGHT / 1080
ORG = (7 * WND_WIDTH // 8, WND_HEIGHT // 2)

# import game_over image
GAME_OVER = cv2.imread(f'images/{settings.THEME}_game_over.png')
GAME_OVER = cv2.resize(GAME_OVER, (WND_WIDTH, WND_HEIGHT))
