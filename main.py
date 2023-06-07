from random import randint as randi

from overlap import *


def main():
    # initialize game logic variables
    period = PERIOD
    speed = SPEED
    score = 0
    num_frames = 0

    # initialize ball logic
    ball_i = 0  # used for choosing the ball that will be created
    ball_img_i = 0  # used for choosing ball image (according to rotation)
    balls = []  # list of balls (coordinates)
    for _ in range(NUM_BALLS):  # initialize balls outside the frame
        balls.append([-BALL_SIZE, -BALL_SIZE])

    # create full screen window
    cv2.namedWindow('Face Basketball', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Face Basketball', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow('Face Basketball', WND_WIDTH, WND_HEIGHT)

    while True:
        _, frame = cap.read()  # read the captured video frames
        num_frames += 1

        frame = cv2.flip(frame, 1)  # the frames captured are by default mirrored, so they need to be flipped
        # converting the image to grayscale makes it easier for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)  # detect faces in frame

        # create ball at random location
        if not num_frames % (period // 10):
            if balls[ball_i][1] == -BALL_SIZE:  # if ball is outside the frame create it
                balls[ball_i][0] = -BALL_SIZE  # set ball height to maximum (over the upper edge)
                # set random horizontal position between margins
                balls[ball_i][1] = randi(MARGIN, WIDTH - BALL_SIZE - MARGIN)
                ball_i += 1
                ball_i %= NUM_BALLS
                speed += SPEED_INCREASE  # increase the speed of the balls
                period -= PERIOD_DECREASE if period > 10 else 0  # decrease the period between balls

        # move the balls downwards and check if any reached the bottom edge of the frame
        if not move_balls(frame, balls, ball_img_i, speed):
            cv2.imshow('Face Basketball', GAME_OVER)  # show game over image
            while True:
                k = cv2.waitKey(0) & 0xff  # wait for user input
                if k == 27:  # if ESC is pressed close game
                    return False
                if k == 13:  # if ENTER is pressed restart game
                    return True

        # increment ball_img_i to rotate the ball in the next frame
        ball_img_i += 1
        ball_img_i %= 18

        for face in faces:
            overlap_face(frame, face)  # overlap net image over each face
            score += check_goal(face, balls)  # check if any faces (nets) have scored

        # create game frame starting with a blank frame and adding the processed frame inside it
        wnd_frame = BLANK_FRAME.copy()
        wnd_frame[DIFF_HEIGHT: WND_HEIGHT - DIFF_HEIGHT, DIFF_WIDTH: WND_WIDTH - DIFF_WIDTH, :] = frame

        # add score to the frame
        text = f"{score}".zfill(3)
        wnd_frame = cv2.putText(img=wnd_frame, text=text, org=ORG, fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                                fontScale=FONT_SCALE, thickness=2, color=(0, 0, 255))

        # show frame
        cv2.imshow('Face Basketball', wnd_frame)

        # wait for Esc key to stop
        k = cv2.waitKey(20) & 0xff
        if k == 27:
            while True:
                # show pause menu
                cv2.imshow('Face Basketball', cv2.addWeighted(wnd_frame, 1, PAUSE, 0.5, 0))
                k = cv2.waitKey(0) & 0xff  # wait for user input
                if k == 27 or k == 13:  # if ESC or ENTER is pressed resume game
                    break
                elif k == 113:  # if Q is pressed quit game
                    return False


# launch game
while main():
    pass

# close the window and de-allocate any associated memory usage
cap.release()
cv2.destroyAllWindows()
