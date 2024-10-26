import cv2
import numpy as np
from emotion_recognition import EmotionDetector
from music_player import MusicPlayer
import tkinter as tk
from tkinter import messagebox
import random
from emotion_songs import emotion_songs
import os
import pygame
import webbrowser

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(fill=tk.BOTH, expand=True)
        self.create_widgets()
        self.emotion_detector = EmotionDetector('models/cnn_model.h5', {
            0: "Angry", 1: "Happy", 2: "Neutral",
            3: "Sad", 4: "Surprised"
        })
        self.music_player = MusicPlayer()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.running = False

    def create_widgets(self):
        #window size
        self.master.geometry("800x600")

        # Title Label
        self.title_label = tk.Label(self, text="情绪检测音乐推送", font=("Arial", 24))
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20)

        # Buttons
        self.detect_button = tk.Button(self, text="开始情绪检测", command=self.start_detection, font=("Arial", 16))
        self.detect_button.grid(row=1, column=0, pady=10)

        self.stop_button = tk.Button(self, text="停止情绪检测", command=self.stop_detection, font=("Arial", 16))
        self.stop_button.grid(row=2, column=0, pady=10)

        self.stop_music_button = tk.Button(self, text="停止播放音乐", command=self.stop_music, font=("Arial", 16))
        self.stop_music_button.grid(row=3, column=0, pady=10)

        # Emotion Labels
        self.emotion_label = tk.Label(self, text="预测情绪:", font=("Arial", 16))
        self.emotion_label.grid(row=4, column=0, pady=10)

        self.angry_label = tk.Label(self, text="Angry: 0.0", font=("Arial", 14))
        self.angry_label.grid(row=5, column=0, sticky="w")

        self.happy_label = tk.Label(self, text="Happy: 0.0", font=("Arial", 14))
        self.happy_label.grid(row=6, column=0, sticky="w")

        self.neutral_label = tk.Label(self, text="Neutral: 0.0", font=("Arial", 14))
        self.neutral_label.grid(row=7, column=0, sticky="w")

        self.sad_label = tk.Label(self, text="Sad: 0.0", font=("Arial", 14))
        self.sad_label.grid(row=8, column=0, sticky="w")

        self.surprised_label = tk.Label(self, text="Surprised: 0.0", font=("Arial", 14))
        self.surprised_label.grid(row=9, column=0, sticky="w")

        # Recommendation Frame
        self.recommend_frame = tk.Frame(self, width=600, height=400, bd=2, relief=tk.SUNKEN, bg="#F0F0F0")
        self.recommend_frame.grid(row=1, column=1, rowspan=9, padx=20, pady=20, sticky="nsew")

        self.recommend_label = tk.Label(self.recommend_frame, text="推荐歌曲:", font=("Arial", 16), bg="#F0F0F0")
        self.recommend_label.pack(side="top", pady=10)

        self.recommend_text = tk.Label(self.recommend_frame, text="请等待情绪检测...", font=("Arial", 14), bg="#F0F0F0")
        self.recommend_text.pack(side="top", pady=10)
        self.song_links = []  # Store song link labels for updating
        for _ in range(2):  # Create two labels for recommendations
            label = tk.Label(self.recommend_frame, text="", font=("Arial", 14), fg="blue", cursor="hand2")
            label.pack(side="top", pady=2)
            label.bind("<Button-1>", self.open_link)
            self.song_links.append(label)
        # Recommendation Frame
        self.recommend_frame = tk.Frame(self, bd=2, relief=tk.SUNKEN, bg="#F0F0F0")
        self.recommend_label = tk.Label(self.recommend_frame, text="推荐歌曲:", font=("Arial", 16), bg="#F0F0F0")
        self.recommend_frame.grid(row=1, column=1, rowspan=9, padx=20, pady=20, sticky="nsew")
        self.recommend_label.grid(row=0, column=0, pady=10, sticky="nsew")

        # 将weight设置为0，使得顶部的标签不会被拉伸
        self.recommend_frame.grid_rowconfigure(0, weight=0)  
        self.recommend_frame.grid_rowconfigure(1, weight=0)  
        self.recommend_frame.grid_rowconfigure(2, weight=1)  
        self.recommend_frame.grid_columnconfigure(0, weight=1) 

        self.recommend_text = tk.Label(self.recommend_frame, text="请等待情绪检测...", font=("Arial", 14), bg="#F0F0F0", wraplength=500)
        self.recommend_text.grid(row=1, column=0, pady=10, sticky="nsew")
        self.song_links_frame = tk.Frame(self.recommend_frame, bg="#F0F0F0")
        self.song_links_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        self.song_links = []  
        for _ in range(2):  
             label = tk.Label(self.song_links_frame, text="", font=("Arial", 14), fg="blue", cursor="hand2", wraplength=400)
             label.pack(side="top", pady=2, fill="x")
             label.bind("<Button-1>", self.open_link)
             self.song_links.append(label)

        # Quit Button
        self.quit = tk.Button(self, text="QUIT", fg="red", command=self.on_quit, font=("Arial", 16))
        self.quit.grid(row=10, column=0, columnspan=2, pady=20)

    def on_quit(self):
        self.stop_detection()
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.music.stop()
        self.master.destroy()

    def start_detection(self):
        self.running = True
        self.detect_button["state"] = "disabled"
        self.detect_emotion()

    def stop_detection(self):
        self.running = False
        self.detect_button["state"] = "normal"
        cv2.destroyAllWindows()

        # Get detected emotion
        detected_emotion = self.emotion_detector.max_emotion

        # Update recommendation text
        if detected_emotion!=None:
           self.recommend_text.config(text=f"检测到您当前的情绪是 {detected_emotion}，为您播放一首歌，除此还推荐您听下面两首歌")

        # Select songs for detected emotion
        if detected_emotion in emotion_songs: 
            recommended_songs = random.sample(emotion_songs[detected_emotion], 2)
            for i, song in enumerate(recommended_songs):           
                song_details = song.rsplit(' ', 1)
                song_name_artist, song_url = song_details[0], song_details[1]
                self.song_links[i].config(text=song_name_artist)
                self.song_links[i].link = song_url  
        else:
            for label in self.song_links:
                label.config(text="")
                label.link = ""

        self.music_player.play_music(self.emotion_detector.max_emotion)

    def open_link(self, event):
        webbrowser.open_new(event.widget.link)

    def stop_music(self):
        pygame.mixer.music.stop()
        messagebox.showinfo("提示", "音乐播放已停止")

    def detect_emotion(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            cv2.destroyAllWindows()
            return

        frame, emotion_dict = self.emotion_detector.detect_emotion(frame)
        cv2.imshow('Video', frame)

        if emotion_dict:
            max_emotion = max(emotion_dict, key=emotion_dict.get)
            self.emotion_detector.max_emotion = max_emotion
            self.angry_label["text"] = f"Angry: {emotion_dict['Angry']:.2f}"
            self.happy_label["text"] = f"Happy: {emotion_dict['Happy']:.2f}"
            self.neutral_label["text"] = f"Neutral: {emotion_dict['Neutral']:.2f}"
            self.sad_label["text"] = f"Sad: {emotion_dict['Sad']:.2f}"
            self.surprised_label["text"] = f"Surprised: {emotion_dict['Surprised']:.2f}"

        self.master.after(10, self.detect_emotion)

root = tk.Tk()
root.title("22121105尹俊杰情绪检测音乐推送")
app = Application(master=root)
app.mainloop()