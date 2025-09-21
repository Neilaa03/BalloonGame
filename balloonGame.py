import pygame
import random
import cv2
import os
import numpy as np


# //// HELPER FUNCTIONS /////
def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]
    H, W = background.shape[:2]

    # Clip overlay if it goes outside the background
    x_start = max(x, 0)
    y_start = max(y, 0)
    x_end = min(x + w, W)
    y_end = min(y + h, H)

    overlay_x_start = x_start - x
    overlay_y_start = y_start - y
    overlay_x_end = overlay_x_start + (x_end - x_start)
    overlay_y_end = overlay_y_start + (y_end - y_start)

    if x_start >= x_end or y_start >= y_end:
        return background  # completely off-screen

    roi = background[y_start:y_end, x_start:x_end]

    # Extract the relevant part of the overlay
    overlay_part = overlay[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end]

    # Separate alpha channel and normalize
    alpha = overlay_part[:, :, 3:] / 255.0
    alpha_inv = 1.0 - alpha

    # Blend
    roi[:] = alpha_inv * roi + alpha * overlay_part[:, :, :3]

    return background



def reset_game():
    global game_running, game_over, balloons, frame_count, score, losses
    game_running = True
    game_over = False
    balloons = []
    frame_count = 0
    score = 0
    losses = 0


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit(" Could not open webcam")

#color range
pink_lower = np.array([140, 100, 100], np.uint8)
pink_upper = np.array([170, 255, 255], np.uint8)

min_area = 400
max_area = 5000

center = None
canvas = None
prev_point = None

alpha = 0.2 #smoothing factor

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)
colors = [red, green, blue, white]
color_pos = [0, 0, 0, 0]  # x positions for color rectangles
current_color = white  # Default 

#BALLOON GAME PARAMS
game_running = False
game_over = False
balloons = []
frame_count = 0
MIN_SPEED = 5
MAX_SPEED = 10
FLOWN_RATE = 30  # frames per balloon
score = 0       #red: 10p , white: 2p, blue: 5p, green: 8p
losses = 0
MAX_LOSSES = 10

#balloon : {x, y, radius, color, speed}

FRAME_WIDTH = int(cap.get(3))
FRAME_HEIGHT = int(cap.get(4))
#BG images
if not os.path.exists('assets/img/background.jpg'):
    raise SystemExit(" Could not find background image")
bg = cv2.imread('assets/img/background.jpg')
#bg = cv2.resize(bg, (FRAME_WIDTH, FRAME_HEIGHT))


# //////////////// ASSETS ///////////////////////

pygame.mixer.init()

if not os.path.exists('assets/sounds/soundTrack.mp3'):
    raise SystemExit(" Could not find sound file")
bg_music = "assets/sounds/soundTrack.mp3"
#pop_sound = pygame.mixer.Sound('assets/sounds/pop.wav')
#loss_sound = pygame.mixer.Sound('assets/sounds/loss.wav')

pygame.mixer.music.load(bg_music)
pygame.mixer.music.set_volume(0.2)
pygame.mixer.music.play(loops=-1)

#balloon images 
red_balloon = cv2.imread('assets/img/balloons/red.png', cv2.IMREAD_UNCHANGED)
green_balloon = cv2.imread('assets/img/balloons/green.png', cv2.IMREAD_UNCHANGED)
blue_balloon = cv2.imread('assets/img/balloons/blue.png', cv2.IMREAD_UNCHANGED)
white_balloon = cv2.imread('assets/img/balloons/white.png', cv2.IMREAD_UNCHANGED)

red_balloon = cv2.resize(red_balloon, (60, 80))
green_balloon = cv2.resize(green_balloon, (60, 80))
blue_balloon = cv2.resize(blue_balloon, (60, 80))
white_balloon = cv2.resize(white_balloon, (60, 80))

#se img shapes 
print(red_balloon.shape)
print(green_balloon.shape)
print(blue_balloon.shape)
print(white_balloon.shape)

while True:
    ret, frame = cap.read()    

    if not ret:
        print(" Can't receive frame. Exiting ...")
        break
    
    if canvas is None:
        canvas = np.zeros_like(frame)


    # bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
    # frame = cv2.addWeighted(frame, 0.2, bg, 1, 0)

    frame = cv2.flip(frame, 1)        # Flip the frame horizontally
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    mask = cv2.inRange(hsv, pink_lower, pink_upper)

    key = cv2.waitKey(1) & 0xFF
    if not game_running or game_over:
        cv2.putText(frame, 'Press S to Start the Balloon Game or Q to quit', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Webcam', frame)       # Show the frame
        #key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            reset_game()
        elif key == ord('q'):
            break
        
        continue

    #clean mask
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    #score display
    cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Losses: {losses}/{MAX_LOSSES}', (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)

    #generate balloons
    if frame_count % FLOWN_RATE == 0:  # Every 30 frames, add a new balloon
        balloon = {
            'x': random.randint(50, 590),
            'y': 480,
            'radius': random.randint(20, 40),
            'color': random.choice(colors),
            'speed': random.uniform(MIN_SPEED, MAX_SPEED),
            'popped': False
        }
        balloons.append(balloon)


    # Update and draw balloons
    for b in balloons:
        b['y'] -= b['speed']
        if b['y'] + b['radius'] > 0:  # Only draw if above the bottom of the frame
            #cv2.circle(frame, (int(b['x']), int(b['y'])), b['radius'], b['color'], -1)
            if b['color'] == red:
                frame = overlay_image(frame, red_balloon, int(b['x']) - 30, int(b['y']) - 80)
            elif b['color'] == green:
                frame = overlay_image(frame, green_balloon, int(b['x']) - 30, int(b['y']) - 80)
            elif b['color'] == blue:
                frame = overlay_image(frame, blue_balloon, int(b['x']) - 30, int(b['y']) - 80)
            elif b['color'] == white:
                frame = overlay_image(frame, white_balloon, int(b['x']) - 30, int(b['y']) - 80)



    #create contour
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #draw disk at the center
        if area > 400:
            
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                if center is None:
                    center = (cX, cY)
                else:
                    center = (
                        int(center[0] * (1 - alpha) + cX * alpha),
                        int(center[1] * (1 - alpha) + cY * alpha)
                    )



                cv2.circle(frame, (cX, cY), 10, current_color, -1)  # Draw a white disk
                prev_point = center
            else:
                center = None
                prev_point = None

    #pop balloon
    if center is not None:
        for b in balloons:
            dx = b['x'] - center[0]
            dy = b['y'] - center[1]
            distance = (dx**2 + dy**2)**0.5
            if distance < b['radius'] + 10:  # 10 is the radius of the drawing point
                balloons.remove(b)

                #update score
                if b['color'] == red:
                    score += 10
                elif b['color'] == green:
                    score += 8
                elif b['color'] == blue:
                    score += 5
                elif b['color'] == white:
                    score += 2
                
                #add pop sound here
                break

    frame_count += 1

    #update balloon list
   # balloons = [b for b in balloons if b['y'] + b['radius'] > 0]
    for b in balloons[:]:
        if b['y'] + b['radius'] < 0:
            if not b['popped']:
                losses += 1
            balloons.remove(b)
            

    if (frame_count % 200) == 0:
        MIN_SPEED += 5
        MAX_SPEED += 5
        FLOWN_RATE = max(10, FLOWN_RATE - 2)  # Decrease the rate to a minimum of 10 frames
    

    if losses >= MAX_LOSSES:
        game_over = True
        game_running = False

        cv2.putText(frame, 'Game Over! Press Q to Quit', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Webcam', frame)       # Show the frame
        key = cv2.waitKey(1) & 0xFF
        game_running = False
        balloons = []
        if key == ord('q'):
            break
        continue

    # Combine the frame and the canvas
    res = cv2.add(frame, canvas)
    cv2.imshow('Webcam', res)       # Show the combined result

cap.release()                          # Release the webcam
cv2.destroyAllWindows()                # Close the window


#TODO 
#balloons are colored images of ballons selected randomly by color
#add background image
#add pop up sound and loss sound
#