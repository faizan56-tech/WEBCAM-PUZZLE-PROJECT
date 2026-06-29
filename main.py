import cv2
import mediapipe as mp
import numpy as np
import math
import random

# 1. MediaPipe Hands initialize karein
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_drawing = mp.solutions.drawing_utils

# 2. Webcam start karein
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Box/Frame ki positions
box_x1, box_y1 = 150, 100
box_x2, box_y2 = 450, 400
box_width = box_x2 - box_x1
box_height = box_y2 - box_y1

# Puzzle state variables
is_captured = False
grid_pieces = []
selected_piece_idx = None  # Swap ke liye select kiya gaya piece

print("Final Step Active: Shuffled pieces ko ungli se swap karein!")

def create_puzzle_grid(image):
    pieces = []
    p_width = box_width // 3
    p_height = box_height // 3
    for r in range(3):
        for c in range(3):
            piece = image[r*p_height:(r+1)*p_height, c*p_width:(c+1)*p_width].copy()
            pieces.append(piece)
    random.shuffle(pieces)
    return pieces

def get_piece_index_at_point(x, y):
    """Batata hai ke cursor/finger konsi grid position (0-8) par hai"""
    if box_x1 < x < box_x2 and box_y1 < y < box_y2:
        p_width = box_width // 3
        p_height = box_height // 3
        c = (x - box_x1) // p_width
        r = (y - box_y1) // p_height
        return int(r * 3 + c)
    return None

def draw_grid(frame, pieces):
    p_width = box_width // 3
    p_height = box_height // 3
    idx = 0
    for r in range(3):
        for c in range(3):
            x_pos = box_x1 + (c * p_width)
            y_pos = box_y1 + (r * p_height)
            
            if idx < len(pieces):
                frame[y_pos:y_pos+p_height, x_pos:x_pos+p_width] = pieces[idx]
            
            # Agar koi piece select ho rakha hai, toh usko highlight kar dena (Yellow border)
            if idx == selected_piece_idx:
                cv2.rectangle(frame, (x_pos, y_pos), (x_pos+p_width, y_pos+p_height), (0, 255, 255), 4)
            else:
                cv2.rectangle(frame, (x_pos, y_pos), (x_pos+p_width, y_pos+p_height), (255, 255, 255), 1)
            idx += 1

# Pervious state tracking for pinch-and-drag swap
was_pinching = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    box_color = (0, 0, 255)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_x = int(hand_landmarks.landmark[8].x * w)
            index_y = int(hand_landmarks.landmark[8].y * h)
            thumb_x = int(hand_landmarks.landmark[4].x * w)
            thumb_y = int(hand_landmarks.landmark[4].y * h)

            if box_x1 < index_x < box_x2 and box_y1 < index_y < box_y2:
                box_color = (0, 255, 0)
                distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
                
                # Pinch Detection for Logic
                if distance < 30:
                    if not is_captured:
                        # Capture / Scramble state
                        raw_captured = frame[box_y1:box_y2, box_x1:box_x2].copy()
                        grid_pieces = create_puzzle_grid(raw_captured)
                        is_captured = True
                        print("Puzzle Scrambled!")
                    else:
                        # Swap mechanism (Pinch karke piece select/swap karna)
                        if not was_pinching:
                            clicked_idx = get_piece_index_at_point(index_x, index_y)
                            if clicked_idx is not None:
                                if selected_piece_idx is None:
                                    # Pehla piece select kia
                                    selected_piece_idx = clicked_idx
                                    print(f"Piece {selected_piece_idx} selected.")
                                else:
                                    # Doosri jagah pinch kia, toh swap kardein!
                                    # Dono indices aapas mein swap ho jayenge
                                    idx1, idx2 = selected_piece_idx, clicked_idx
                                    grid_pieces[idx1], grid_pieces[idx2] = grid_pieces[idx2], grid_pieces[idx1]
                                    print(f"Swapped {idx1} and {idx2}.")
                                    selected_piece_idx = None # Selection clear
                            was_pinching = True
                else:
                    was_pinching = False

            cv2.circle(frame, (index_x, index_y), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (thumb_x, thumb_y), 8, (0, 255, 255), cv2.FILLED)

    if is_captured and grid_pieces:
        draw_grid(frame, grid_pieces)
        cv2.putText(frame, "Swap pieces to solve!", (150, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 3)

    if not is_captured:
        cv2.putText(frame, "Make a frame & Pinch to scramble", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Webcam Puzzle - Final", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("Webcam Puzzle - Final", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()