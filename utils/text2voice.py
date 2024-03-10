import pygame
from gtts import gTTS
def save_vocie(text,file_name='vocie.mp3'):
    tts = gTTS(text=text, lang='en-GB')
# 识别正确返回语音二进制 错误则返回dict
    tts.save(file_name)

def voice_show(file_name):
    pygame.mixer.init()
    track = pygame.mixer.music.load(file_name)
    pygame.mixer.music.play()