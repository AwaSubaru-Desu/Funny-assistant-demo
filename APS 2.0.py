import cv2
import numpy as np
import mediapipe as mp
import random
import time
import sys
from gtts import gTTS
import pygame.mixer
import os
import tempfile
import speech_recognition as sr
import threading

# 初始化 Mediapipe 姿态检测器
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 初始化摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Please check if the camera is available.")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Playful suggestions pool (grouped from easy to hard)
pose_suggestions_groups = [
    ["Close your left eye!", "Give a big laugh!", "Raise your right hand!", "Put your left hand on your hip!", "Stand on one leg!"],
    ["Wave both hands!", "Jump up!", "Make a funny face!", "Touch your toes!", "Do a spin!"],
    ["Pretend to be a tree!", "Do a superhero landing!", "Look surprised!", "Pretend to hold an invisible ball!", "Do a yoga pose!"],
    ["Clap your hands above your head!", "Pretend you're holding a heavy box!", "Balance on your tiptoes!", "Do a side lunge!", "Cross your arms and look serious!"],
    ["Pretend you're riding a bicycle!", "Stretch both arms out and rotate slowly!", "Do a martial arts stance!", "Crouch down and look up!", "Pretend to swim in place!"],
    ["Act like you're running in slow motion!", "Do a karate kick!", "Jump and reach for the sky!", "Do the victory V pose with both hands!", "Make a heart shape with your hands!"],
    ["Stand like a superhero!", "Pretend to be a bird flying!", "Do a funny dance!", "Do a side plank!", "Pretend to hold a giant ball over your head!"]
]
# 缓存音频文件
def cache_audio_files():
    audio_folder = os.path.join(tempfile.gettempdir(), "audio_prompts")
    os.makedirs(audio_folder, exist_ok=True)
    audio_paths = {}
    for i, group in enumerate(pose_suggestions_groups):
        for j, suggestion in enumerate(group):
            filename = os.path.join(audio_folder, f"audio_{i}_{j}.mp3")
            if not os.path.exists(filename):
                tts = gTTS(text=suggestion, lang='en')
                tts.save(filename)
            audio_paths[(i, j)] = filename
    return audio_paths

pygame.mixer.init()
audio_files = cache_audio_files()

def play_audio(filename):
    def play():
        try:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Error playing audio: {e}")
    threading.Thread(target=play).start()

def display_text(frame, text, position, color=(0, 255, 0), font_scale=1, thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def grade_photo():
    score = random.randint(0, 100)
    feedback = f"Score: {score}. {'Amazing!' if score > 70 else 'Good try, keep going!'}"
    return score, feedback

def resize_with_aspect_ratio(image, width, height):
    h, w = image.shape[:2]
    aspect_ratio = w / h
    new_w, new_h = width, int(width / aspect_ratio)
    if new_h > height:
        new_h = height
        new_w = int(height * aspect_ratio)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top = (height - new_h) // 2
    bottom = height - new_h - top
    left = (width - new_w) // 2
    right = width - new_w - left
    return cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

print("AI Photo Assistant is running. Press 'q' to quit.")

last_suggestion_time = 0
suggestion_interval = 7
suggestion_group_index = random.randint(0, len(pose_suggestions_groups) - 1)
suggestion_index = 0
suggestions_given = 0
photos = []
current_suggestion = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if current_suggestion:
        display_text(frame, current_suggestion, (50, 100), color=(255, 0, 0))

    current_time = time.time()
    if current_time - last_suggestion_time > suggestion_interval:
        current_suggestion = pose_suggestions_groups[suggestion_group_index][suggestion_index]
        audio_filename = audio_files[(suggestion_group_index, suggestion_index)]
        play_audio(audio_filename)

        capture_time = time.time() + 4
        while time.time() < capture_time:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            display_text(frame, "Hold your pose...", (50, 100), color=(0, 255, 0))
            cv2.imshow('AI Photo Assistant', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        ret, photo_frame = cap.read()
        if ret:
            photo_frame = cv2.flip(photo_frame, 1)
            photos.append(photo_frame)

        suggestion_index += 1
        suggestions_given += 1
        last_suggestion_time = current_time

    if suggestions_given == 5:
        grid_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        positions = [(0, 426), (240, 0), (240, 426), (240, 852), (480, 426)]

        for idx, photo in enumerate(photos):
            resized_photo = resize_with_aspect_ratio(photo, 426, 240)
            if resized_photo.shape == (240, 426, 3):
                x, y = positions[idx]
                if y + 240 <= grid_image.shape[0] and x + 426 <= grid_image.shape[1]:
                    grid_image[y:y+240, x:x+426] = resized_photo
                else:
                    print(f"Skipping photo: Position out of range ({x}, {y})")

        score, feedback = grade_photo()
        display_text(grid_image, feedback, (490, 50), color=(0, 255, 255), font_scale=1.5, thickness=3)

        while True:
            cv2.imshow('Photo Summary', grid_image)
            play_audio(feedback)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        suggestion_group_index = random.randint(0, len(pose_suggestions_groups) - 1)
        suggestion_index = 0
        suggestions_given = 0
        photos = []
        current_suggestion = ""

    cv2.imshow('AI Photo Assistant', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
