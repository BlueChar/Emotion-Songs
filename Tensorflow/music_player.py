import pygame
import os
import random
import threading

class MusicPlayer:
    def __init__(self):
        pygame.mixer.init()

    def play_music(self, emotion):
        song_dir = os.path.join(os.path.dirname(__file__), 'songs', emotion)
        songs = [os.path.join(song_dir, song) for song in os.listdir(song_dir) if song.endswith('.mp3')]
        song_path = random.choice(songs)
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play()

        def wait_for_music_to_finish():
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

        threading.Thread(target=wait_for_music_to_finish).start()