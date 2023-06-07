from init import *


# overlap net image over face
def overlap_face(frame, face):
    xf, yf, wf, lf = face  # get face location
    mask = cv2.resize(NET_IMG, (wf, lf))  # resize net image to fit the face
    alpha = mask[:, :, 3] // 255  # get alpha channel of net image where value is 255 (max) and rescale to 0 and 1
    alpha = np.concatenate((alpha[:, :, None], alpha[:, :, None], alpha[:, :, None]), axis=2)
    mask[:, :, :3] *= alpha  # allow net image to show only where alpha channel is 1 (previously 255)
    frame[yf:yf + wf, xf:xf + lf, :3] *= 1 - alpha  # allow face to show only where net image alpha channel is 0
    # add the computed net image and face to the initial frame
    frame[yf:yf + wf, xf:xf + lf] = cv2.add(frame[yf:yf + wf, xf:xf + lf], mask[:, :, :3])


# change ball locations and overlap their images
def move_balls(frame, balls, img_i, spd):
    # get the right image (right rotation) with its alpha channel
    ball_img = BALL_IMGS[img_i][0]
    alpha_ball = BALL_IMGS[img_i][1]
    for ball in balls:
        ball[0] += int(spd // 10)  # move ball downwards
        yb, xb = ball  # get ball location
        if xb < 0:  # ball is outside the frame
            continue
        if yb < 0:  # ball is not completely inside the frame
            # allow ball image to show only where alpha channel is 1
            # allow background to show only where ball image alpha channel is 0
            frame[0:yb + BALL_SIZE, xb:xb + BALL_SIZE] = cv2.add(frame[0:yb + BALL_SIZE, xb:xb + BALL_SIZE] *
                                                                 (1 - alpha_ball[-yb:BALL_SIZE, :]),
                                                                 ball_img[-yb:BALL_SIZE, :])
        elif yb <= HEIGHT - BALL_SIZE:  # ball is completely inside the frame
            # allow ball image to show only where alpha channel is 1
            # allow background to show only where ball image alpha channel is 0
            frame[yb:yb + BALL_SIZE, xb:xb + BALL_SIZE] = cv2.add(
                frame[yb:yb + BALL_SIZE, xb:xb + BALL_SIZE] * (1 - alpha_ball), ball_img[:, :, :3])
        else:  # ball has reached bottom edge, i.e. player has lost
            return False
    return True  # return true if player has not lost


# check if a ball has entered a net
def check_goal(face, balls):
    xf, yf, wf, lf = face  # get face location
    # specify the center where the ball is considered scored
    xf += lf // 2
    yf += 2 * wf // 3
    # specify maximum distance from the center for the ball to be scored
    dist = BALL_SIZE // 2
    scr = 0  # score addition
    for ball in balls:
        yb, xb = ball  # get ball location
        # get distance of ball from previously specified center
        dx = abs(xf - (xb + BALL_SIZE // 2))
        dy = abs(yf - (yb + BALL_SIZE // 2))
        if dx < dist and dy < dist:  # if ball is within the specified distance from the specified center
            # place the ball outside the frame
            ball[0] = -BALL_SIZE
            ball[1] = -BALL_SIZE
            scr += 1  # add 1 to the score addition
    return scr  # return score addition which will be added to final score
