import asyncio
import speech_recognition as sr
# import keyboard
# import time

async def get_audio():
    # obtain audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something! You have 2 seconds")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, timeout=2)

    # recognize speech using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        print("Google Speech Recognition thinks you said '" + r.recognize_google(audio) + "'")
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


async def print_nothing():
    print("Nothing")
    await asyncio.sleep(1)


while True:
    print("Incomplete")
    # Prototype Push-to-Talk
    # talk = input('Press K for push-to-talk\n')
    # if talk == 'k':
    #    get_audio()
    # else:
    #    print("bingo bongo")
