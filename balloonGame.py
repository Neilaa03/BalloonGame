import pygame
import random
import cv2
import os
import numpy as np
import mediapipe as mp


def save_score(score, filename="scores.txt"):
    with open(filename, "a") as f:
        f.write(str(score) + "\n")

def load_scores(filename="scores.txt"):
    try:
        with open(filename, "r") as f:
            scores = [int(line.strip()) for line in f.readlines()]
        return sorted(scores, reverse=True)
    except FileNotFoundError:
        return []



#// HAND DETECTION ////
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)


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
if not os.path.exists('assets/img/bg.jpeg'):
    raise SystemExit(" Could not find background image")
bg = cv2.imread('assets/img/bg.jpeg')
bg = cv2.resize(bg, (FRAME_WIDTH, FRAME_HEIGHT))


# //////////////// ASSETS ///////////////////////

pygame.mixer.init()

if not os.path.exists('assets/sounds/soundTrack.mp3'):
    raise SystemExit(" Could not find sound file")
bg_music = "assets/sounds/soundTrack.mp3"
pop_sound = pygame.mixer.Sound('assets/sounds/popSound.wav')
#loss_sound = pygame.mixer.Sound('assets/sounds/loss.wav')

pygame.mixer.music.load(bg_music)
pygame.mixer.music.set_volume(0.2)
pygame.mixer.music.play(loops=-1)

#balloon images 
balloon_colors = ['red', 'green', 'blue', 'white']
balloons_imgs = {}

for color in balloon_colors:
    img = cv2.imread(f'assets/img/balloons/{color}.png', cv2.IMREAD_UNCHANGED)
    balloons_imgs[color] = cv2.resize(img, (60, 80))


while True:
    ret, frame = cap.read()    

    if not ret:
        print(" Can't receive frame. Exiting ...")
        break
    
    frame = cv2.flip(frame, 1)  # Flip horizontally
    frame_original = frame.copy() 

    if canvas is None:
        canvas = np.zeros_like(frame)


    bg_resized = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
    bg_alpha = 0.2  # Transparency factor
    frame_display = cv2.addWeighted(frame, bg_alpha, bg_resized, 1 - bg_alpha, 0)


    # HAND DETECTION
    rgb_frame = cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    key = cv2.waitKey(1) & 0xFF
    if not game_running or game_over:
        overlay = frame_display.copy()

        # Semi-transparent black box for title/instructions
        cv2.rectangle(overlay, (50, 120), (frame_display.shape[1] - 50, 220), (0, 0, 0), -1)
        frame_display = cv2.addWeighted(overlay, 0.6, frame_display, 0.4, 0)

        # Title text
        cv2.putText(frame_display, 'Balloon Pop!', (120, 170),
                    cv2.FONT_HERSHEY_TRIPLEX, 2, (128, 10, 128), 3, cv2.LINE_AA)

        # Instruction text (centered)
        cv2.putText(frame_display, 'Press S to Start or Q to Quit',
                    (100, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Webcam', frame_display)
        if key == ord('s'):
            reset_game()
        elif key == ord('q'):
            break
        
        continue

    # ---- Score & Losses with background box ----
    cv2.rectangle(frame_display, (5, 5), (250, 45), (0, 0, 0), -1)  # Black box
    cv2.putText(frame_display, f'Score: {score}', (15, 35),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.rectangle(frame_display, (380, 5), (640, 45), (0, 0, 0), -1)
    cv2.putText(frame_display, f'Losses: {losses}/{MAX_LOSSES}', (390, 35),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    

    # -----Score keys-----
    frame_x = 20
    frame_y = 55
    frame_size = 130
    overlay = frame_display.copy()

    cv2.rectangle(overlay, (frame_x, frame_y),
              (frame_x + frame_size, frame_y + 100), (0, 0, 0), -1)
    frame_display = cv2.addWeighted(overlay, 0.4, frame_display, 0.6, 0)

    cv2.putText(frame_display, "Red = 10", (frame_x + 10, frame_y + 15),
            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,255), 1)
    cv2.putText(frame_display, "Green = 8", (frame_x + 10, frame_y + 35),
            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,0), 1)
    cv2.putText(frame_display, "Blue = 5", (frame_x + 10, frame_y + 55),
            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,0,0), 1)
    cv2.putText(frame_display, "White = 2", (frame_x + 10, frame_y + 75),
            cv2.FONT_HERSHEY_DUPLEX, 0.6, (200,200,200), 1)


    #clean mask
    hsv = cv2.cvtColor(frame_display, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, pink_lower, pink_upper)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)



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
        if b['y'] + b['radius'] > 0:
        # Overlay image using the balloon's color
            color_name = None
            if b['color'] == red:
                color_name = 'red'
            elif b['color'] == green:
                color_name = 'green'
            elif b['color'] == blue:
                color_name = 'blue'
            elif b['color'] == white:
                color_name = 'white'

            if color_name:
                frame_display = overlay_image(frame_display, balloons_imgs[color_name], int(b['x']) - 30, int(b['y']) - 80)


    center = None
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Example: use the tip of the index finger (landmark 8)
            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)
            center = (x, y)

            # Draw hand landmarks (optional, for debugging)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw a small circle where the finger is
            cv2.circle(frame, center, 10, (0, 255, 0), -1)
            


    #pop balloon
    if center is not None:
        for b in balloons:
            dx = b['x'] - center[0]
            dy = b['y'] - center[1]
            distance = (dx**2 + dy**2)**0.5
            if distance < b['radius'] + 10:  # 10 is the radius of the drawing point
                balloons.remove(b)
                pop_sound.play()
                b['popped'] = True

                #update score
                if b['color'] == red:
                    score += 8
                elif b['color'] == green:
                    score += 6
                elif b['color'] == blue:
                    score += 3
                elif b['color'] == white:
                    score += 1                
                break

    frame_count += 1

    #update balloon list
    for b in balloons[:]:
        if b['y'] + b['radius'] < 0:
            if not b['popped']:
                losses += 1
            balloons.remove(b)
            

    if (frame_count % 200) == 0:
        MIN_SPEED += 6
        MAX_SPEED += 6
        FLOWN_RATE = max(10, FLOWN_RATE - 5)  # Decrease the rate to a minimum of 10 frames
    

    if losses >= MAX_LOSSES:
        game_over = True
        game_running = False

        save_score(score)
        high_scores = load_scores()

        cv2.putText(frame_display, 'GAME OVER!', (150, 200),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)

         # High scores
        for i, s in enumerate(high_scores[:5]):
            cv2.putText(frame_display, f"{i+1}. {s}", (200, 220 + i*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Restart/Quit message LOWER on the screen (not overwriting)
        cv2.putText(frame_display, 'Press Q to Quit or S to Restart',
                    (80, 480), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        cv2.imshow('Webcam', frame_display)       # Show the frame
        key = cv2.waitKey(1) & 0xFF
       # game_running = False
        balloons = []
        if key == ord('q'):
            break
        continue

    
    # Allow quitting anytime during gameplay
    if key == ord('q'):
        game_over = True
        game_running = False
        continue
    
    res = cv2.add(frame_display, canvas)
    cv2.imshow('Webcam', res)       # Show the combined result

cap.release()                          # Release the webcam
cv2.destroyAllWindows()                # Close the window






    # #create contour for pensil detection
    #    # Convert to HSV color space
    # 
    

    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     #draw disk at the center
    #     if area > 400:
            
    #         M = cv2.moments(cnt)
    #         if M["m00"] != 0:
    #             cX = int(M["m10"] / M["m00"])
    #             cY = int(M["m01"] / M["m00"])

    #             if center is None:
    #                 center = (cX, cY)
    #             else:
    #                 center = (
    #                     int(center[0] * (1 - alpha) + cX * alpha),
    #                     int(center[1] * (1 - alpha) + cY * alpha)
    #                 )


    #             cv2.circle(frame, (cX, cY), 10, current_color, -1)  # Draw a white disk
    #             prev_point = center
    #         else:
    #             center = None
    #             prev_point = None