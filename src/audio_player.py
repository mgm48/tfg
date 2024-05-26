import pygame
import time
import os
import asyncio
import pyttsx3
import soundfile as sf
from mutagen.mp3 import MP3

class AudioManager:

    

    def __init__(self):
        self.engine = pyttsx3.init()
        pygame.mixer.init(frequency=48000, buffer=1024) 

    def play_audio(self, file_path, sleep_during_playback=True, delete_file=False, play_using_music=True):
        """
        Parameters:
        file_path (str): path to the audio file
        sleep_during_playback (bool): means program will wait for length of audio file before returning
        delete_file (bool): means file is deleted after playback (note that this shouldn't be used for multithreaded function calls)
        play_using_music (bool): means it will use Pygame Music, if false then uses pygame Sound instead
        """
        print(f"Playing file with pygame: {file_path}")
        if not pygame.mixer.get_init(): # Reinitialize mixer if needed
            pygame.mixer.init(frequency=48000, buffer=1024) 
        if play_using_music:
            # Pygame Music can only play one file at a time
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
        else:
            # Pygame Sound lets you play multiple sounds simultaneously
            pygame_sound = pygame.mixer.Sound(file_path) 
            pygame_sound.play()

        if sleep_during_playback:
            # Calculate length of the file, based on the file format
            _, ext = os.path.splitext(file_path) # Get the extension of this file
            if ext.lower() == '.wav':
                wav_file = sf.SoundFile(file_path)
                file_length = wav_file.frames / wav_file.samplerate
                wav_file.close()
            elif ext.lower() == '.mp3':
                mp3_file = MP3(file_path)
                file_length = mp3_file.info.length
            else:
                print("Cannot play audio, unknown file type")
                return

            # Sleep until file is done playing
            time.sleep(file_length)

            # Delete the file
            if delete_file:
                # Stop Pygame so file can be deleted
                # Note: this will stop the audio on other threads as well, so it's not good if you're playing multiple sounds at once
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                try:  
                    os.remove(file_path)
                    print(f"Deleted the audio file.")
                except PermissionError:
                    print(f"Couldn't remove {file_path} because it is being used by another process.")


    def config_tts(self,v=0, rate=200,vol=0.7,show_voices_info=False):
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('rate', rate)     # setting up new voice rate
        self.engine.setProperty('volume',vol)   # setting up volume   
        self.engine.setProperty('voice', voices[v].id) # setting voice 
        
        if show_voices_info:
            i = 0
            for voice in voices:
                print("Number:",i)
                print("Voice:",voice.name)
                print(" - ID:",voice.id)
                print(" - Languages:",voice.languages)
                print(" - Gender:",voice.gender)
                print(" - Age:",voice.age)
                print("\n")
                i+=1

    def play_tts(self, path, response):
            filepath = "./" + path + "/res.wav"
            self.engine.save_to_file(response,filepath)
            self.engine.runAndWait()
            self.play_audio(filepath, delete_file=True)




    