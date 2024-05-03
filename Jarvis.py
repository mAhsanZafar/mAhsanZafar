import speech_recognition as sr
import time
import webbrowser
import os
import subprocess
import pyowm
import datetime
from itertools import islice
import pyautogui
import imaplib
import webbrowser
import getpass
import pyttsx3
from screen_brightness_control import set_brightness, get_brightness
import pygetwindow as gw
import win32gui
import win32con
import openai


owm = pyowm.OWM('078b7c5337a253c04dd8db3801bcbb8f') 

def speak(audioString):
    print(audioString)
    engine = pyttsx3.init()

    # Find available voices
    voices = engine.getProperty('voices')

    # Select the voice you want (e.g., 'Microsoft David Desktop')
    desired_voice = 'Microsoft Zira Desktop'

    for voice in voices:
            if desired_voice in voice.name:
                engine.setProperty('voice', voice.id)

    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)

    # Say the message
    engine.say(audioString)
        
    # Wait for the speech to finish
    engine.runAndWait()
    
def greeting():
    # Using pyttsx3 for speech synthesis
    username = getpass.getuser()
    if username == "sa776":
        username = "M Ahsan Zafar"

    message = f"Hey {username}, I'm hare"

    speak(message)

def login():
    greeting()
    voice()

def tell():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("I'm listening...")
        try:
            audio = r.listen(source, timeout=5)  # Add a timeout to prevent indefinite listening
            user_input = r.recognize_google(audio).lower().split(" ")
            print("You said:", user_input)

            if 'simple' in user_input and 'mod' in user_input:
                voice()
            if 'simple' in user_input and 'mode' in user_input:
                voice()
            if 'simple' in user_input and 'mood' in user_input:
                voice()
            else:
                ChatGpt3_5(user_input)
        except sr.UnknownValueError:
            print("Could not understand audio. Please try again.")
            tell()
        except sr.RequestError as e:
            print(f"Error connecting to the speech recognition service: {e}")

    
def voice():
    try:
        # obtain audio from the microphone
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("I'm listening...")
            audio = r.listen(source)
        speech = r.recognize_google(audio)
        print("You said:   " + r.recognize_google(audio))
        speech_search = ['google', 'directions', 'youtube']
        speech = speech.lower().split(" ")
        print(speech)

        # Gets web searches
        if speech[0] in speech_search:
            searching(speech)
            voice()
        elif 'ai' in speech and 'mode' in speech:
            ChatGpt3_5  (speech)
        elif 'ai' in speech and 'mod' in speech:
            ChatGpt3_5  (speech)
        elif 'ai' in speech and 'mood' in speech:
            ChatGpt3_5  (speech)
        # Runs my scripts
        elif "script" and "run" in speech:
            scripts(speech)
            voice()
        # Pause & restart program
        elif 'jarvis' in speech:
            echo(speech)
            voice()
        # Control messaging apps
        elif "send" in speech:
            messaging_app(speech)
            voice()
        elif 'brightness' in speech:
            bright(speech)
            voice()
        elif "task" in speech:
            pyautogui.hotkey('ctrl' ,'shift','esc')
            voice()
        elif "shutdown" in speech:
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
            voice()
        elif "restart" in speech:
            os.system("shutdown /r /t 0")
            voice()
        elif "sleep" in speech:
            os.system("rundll32.exe user32.dll,LockWorkStation")
            voice()
        elif "page up" in speech:
            pyautogui.press('pageup')
            voice()
        elif "page down" in speech:
            pyautogui.press('pagedown')
            voice()
        elif "home" in speech:
            pyautogui.press('home')
            voice()
        elif "end" in speech:
            pyautogui.press('end')
            voice()
        elif "caps" in speech:
            pyautogui.press('capslock')
            voice()
        elif "full" in speech:
            pyautogui.press( 'f11')
            voice()
        elif 'close' in speech:
            if 'yes' in speech:
                pyautogui.hotkey('alt', 'f4')  # Close window
                speak("Window closed.")
                voice()
            close_apps(speech)
            voice()
                # Open applications
        elif 'open' in speech:
            open_apps(speech)
            voice()
        # Provides date information
        elif 'date' in speech:
            date(speech)
            voice()
        # Current time
        elif 'time' in speech:
            speak(datetime.datetime.now().strftime("%I:%M %p"))
            voice()
        # Gets weather data
        elif 'weather' in speech:
            choose_weather(speech)
            voice()
        # Gets temperature data
        elif 'temperature' in speech:
            choose_weather(speech)
            voice()
        # Sunrise time
        elif 'sunrise' in speech:
            choose_weather(speech)
            voice()
        # Sunset time
        elif 'sunset' in speech:
            choose_weather(speech)
            voice()
        # Checks if any new emails have arrived in my inbox
        elif "mail" in speech or "email" in speech:
            check_mail(speech)
            voice()
        elif "minimize" in speech or "maximize" in speech:
            Minimize = win32gui.GetForegroundWindow()
            win32gui.ShowWindow(Minimize, win32con.SW_MINIMIZE)
            voice()
        elif "switch" in speech:
            pyautogui.hotkey('alt', 'tab')  # Switch between open windows
        elif "calander" in speech or "calendar" in speech:
            set_calendar(speech)
        else:
            voice()
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        voice()
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        voice()

def ChatGpt3_5(user_input):
    
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    openai.api_key = openai_api_key

    messages = [
        {"role": "system", "content": "My name is Ahsan."},
    ]

    if 'ai' in user_input and 'mode' in user_input:
        speak("How can I help you?")
        tell()
    elif 'ai' in user_input and 'mood' in user_input:
        speak("How can I help you?")
        tell()
    elif 'ai' in user_input and 'mod' in user_input:
        speak("How can I help you?")
        tell()
    elif user_input:
        user_input_str = " ".join(user_input)  
        messages.append({"role": "user", "content": user_input_str})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        reply = chat.choices[0].message.content
        print("ChatGpt:")
        speak(f" {reply}")
        messages.append({"role": "assistant", "content": reply})
        tell()

'''Putting jarvis to sleep'''
def echo(speech):
    if "sleep" in speech:
        speak("I'm going to nap")
        wake_up = input("Type anything to wake me up...\n")
        if len(wake_up) > 0:
            speak("Hey, I'm awake, do you need anything?")
    else:
        speak("I'm here, what's up?")

'''Main web searches'''
def searching(audio): 
    audio_sentence = " ".join(audio)
    # search google maps
    if "google maps" in audio_sentence:
        #General Maps search
        print("Entering Google Maps search")
        location = audio[2:]
        webbrowser.open('https://www.google.nl/maps/place/' + "+".join(location) + "/&amp;")
        speak("I opened Google Maps for " + " ".join(location))

    #search google
    elif "google" in audio:
        #Google search
        search_phrase = "+".join(audio)
        webbrowser.open('https://www.google.ca/#q=' + search_phrase)
        speak("I opened Google for " + " ".join(search_phrase))

    #full google maps directions from location to destination
    elif "direction from" in audio_sentence:
        #Maps directions from [location] to [destination]
        audio = audio_sentence.split(" ")
        index_to = audio.index("to")
        location = audio[2:index_to]
        destination = audio[index_to + 1:]
        speak_location = " ".join(location)
        location = "+".join(location)
        speak_destination = " ".join(destination)
        destination = "+".join(destination)
        webbrowser.open('https://www.google.ca/maps/dir/' + location + "/" + destination)
        speak("Directions from " + speak_location  + " to " + speak_destination)
    #find directions to google maps destination with location missing
    elif "direction" in audio:
        #Maps directions to destination, requires location
        location = audio[1:]
        location = "+".join(location)
        webbrowser.open('https://www.google.nl/maps/dir//' + location )
        speak("Please enter your current location")
    #searches youtube
    elif "search" in audio:
        print("searching youtube")
        # Searches a youtube video
        search_phrase = audio_sentence.replace("youtube", "").replace("search", "").replace(" ", "+")
        webbrowser.open('https://www.youtube.com/results?search_query=' + search_phrase)
    #play next youtube video
        if "next" in audio:
            print("I'm here")
            pyautogui.hotkey('shift', 'command', 'right')
        # Pause/play youtube videos
        elif "play" in audio or "pause" in audio:
            pyautogui.hotkey('shift', 'command', ' ')

'''Running python scripts'''
def scripts(speech):
    if "soccer" in speech:
        os.system("cd /Users/Add_Folder_Path/ && python3 Soccer_streams.py")
    else:
        os.system("cd /Users/Add_Folder_Path/ && python3 Instalinks.py")

'''Check if any new emails'''
def check_mail(speech):
    # Replace these placeholders with your actual email and password
    email = 'sa7761251@gmail.com'
    password = 'Ahsan@50449'

    obj = imaplib.IMAP4_SSL('imap.gmail.com', '993')
    obj.login(email, password)
    obj.select()
    obj.search(None, 'UnSeen')
    unseen_message = len(obj.search(None, 'UnSeen')[1][0].split()) - 5351
    if unseen_message > 1:
        speak("You have " + str(unseen_message) + " new messages!")
        webbrowser.open('https://mail.google.com')
    else:
        speak("There aren't any new emails!")

    obj = imaplib.IMAP4_SSL('imap.gmail.com', '993')
    obj.login('email@gmail.com', 'password')
    obj.select()
    obj.search(None, 'UnSeen')
    unseen_message = len(obj.search(None, 'UnSeen')[1][0].split()) - 5351
    if unseen_message > 1:
        speak("You have " + str(unseen_message) + " new messages!")
        webbrowser.open('mail.google.com')
    else:
        speak("There isn't any new emails!")

'''Google casting media to ChromeCast'''
def google_cast(speech):
    #format = cast [media] to [monitor or laptop]
    if "monitor" in speech:
        if "youtube" in speech:
            monitor_cast(media='youtube')
        elif "netflix" in speech:
            monitor_cast(media='netflix')
        else:
            monitor_cast(media= speech[1])
    elif "laptop" in speech:
        if "youtube" in speech:
            laptop_cast(media='youtube')
        elif "netflix" in speech:
            laptop_cast(media='netflix')
        else:
            laptop_cast(media=speech[1])

def monitor_cast(media):
    # Assuming Google Chrome is installed on Windows
    subprocess.run(["start", "chrome.exe"], shell=True)
    time.sleep(0.5)
    
    pyautogui.hotkey('ctrl', 'pagedown')  # Assuming this is the Windows shortcut for opening a new tab
    
    time.sleep(0.5)
    pyautogui.hotkey('ctrl', 'l')  # This is to focus on the address bar
    pyautogui.typewrite(media, interval=0.02)
    pyautogui.press('enter')
    time.sleep(0.5)
    
    # Adjust the coordinates based on your screen resolution
    pyautogui.click(x=2150, y=50)
    time.sleep(1.1)
    pyautogui.moveTo(1200, 150)
    pyautogui.moveTo(1200, 160)
    pyautogui.click(x=1200, y=150)
    time.sleep(0.5)
    
    pyautogui.press('esc')
    pyautogui.hotkey('alt', 'tab')  # Assuming this is the Windows shortcut for switching applications

def laptop_cast(media):
    # Cast for 15-inch macbook
    subprocess.call(["/usr/bin/open", "/Applications/Google Chrome.app"])
    time.sleep(0.5)
    pyautogui.hotkey('shift', 'up')
    time.sleep(0.5)
    pyautogui.hotkey('command', 'e')
    pyautogui.typewrite(media, interval=0.02)
    pyautogui.press('enter')
    time.sleep(0.5)
    pyautogui.click(1030, 50)
    time.sleep(1.5)
    pyautogui.moveTo(640, 160)
    pyautogui.click(650, 160)
    time.sleep(0.3)
    pyautogui.press('esc')
    pyautogui.hotkey('command', 'tab')



def messaging_app(speech):
    try:
        if "messenger" in speech:
            if speech[1] == "new":
                receiver = speech[4]
                message = " ".join(speech[5:])
                messenger_automator(receiver, message)
            else:
                message = " ".join(speech[2:])
                messenger_same_convo(message)
        elif "whatsapp" in speech:
            receiver = speech[3]
            message = " ".join(speech[4:])
            whatsapp(receiver, message)
    except IndexError:
        print("Index Error just occurred, repeat what you were saying..")

# New Messenger = send new messenger to [recipient] [message_string]
def messenger_automator(receiver, message):
    # Open the Messenger application
    subprocess.run(["start", "Messenger.exe"], shell=True)  # Update path
    time.sleep(1.5)

    # Automate typing and sending the message
    pyautogui.press('tab', presses=1)
    pyautogui.typewrite(receiver, interval=0.2)
    pyautogui.press('down', presses=1)
    pyautogui.press('enter', presses=1)
    time.sleep(1)
    pyautogui.typewrite(message, interval=0.02)
    time.sleep(0.5)
    pyautogui.hotkey('ctrl', 'enter')  # Adjust for Windows
    speak("Message has been sent to " + receiver)

# Same Messenger = send messenger [message_string]
def messenger_same_convo(message):
    # Open the Messenger application
    subprocess.run(["start", "Messenger.exe"], shell=True)  # Update path
    time.sleep(1)

    # Automate typing and sending the message
    pyautogui.typewrite(message, interval=0.02)
    time.sleep(0.5)
    pyautogui.hotkey('ctrl', 'enter')  # Adjust for Windows

# Message on Whatsapp = send whatsapp to [receiver] [message_string]
def whatsapp(receiver, message):
    subprocess.call(["start", "whatsapp://send?phone=" + receiver])
    time.sleep(1.6)
    pyautogui.press('tab', presses=2, interval=0.5)
    pyautogui.typewrite(message, interval=0.02)
    pyautogui.press('enter')
    time.sleep(1)
    pyautogui.press('tab', presses=1)
    time.sleep(0.4)
    pyautogui.hotkey('command', 'tab')  # Assuming you want to switch back to the previous application
    speak("Whatsapp has been sent to " + receiver)

'''Control Fantastical and set calendar events'''
#set calendar [entry_name] at [location] on the [date] at [time]
def set_calendar(speech):
    if "calendar" in speech:
        pyautogui.hotkey('ctrl', 'c')
        time.sleep(0.2)
        pyautogui.typewrite(" ".join(speech[2:]), interval=0.03)
        pyautogui.press('enter')
        time.sleep(0.7)
        pyautogui.hotkey('ctrl', 'c')
        speak("I have created your calendar event")
    else:
        subprocess.call(["start", "shell:AppsFolder\\Microsoft.WindowsAlarms_8wekyb3d8bbwe!App"])
        time.sleep(1)
        pyautogui.hotkey('ctrl', 'n')
        time.sleep(0.2)
        pyautogui.typewrite(" ".join(speech[2:]), interval=0.02)
        time.sleep(0.1)
        pyautogui.press('enter')
        pyautogui.hotkey('alt', 'tab')  
        speak("I have created a new reminder")

def bright(speech):
    brightness_list = get_brightness()
    current_brightness = brightness_list[0]
    new_brightness = current_brightness  # Initialize new_brightness
    if "increase" in speech:
        if current_brightness <= 90:
            new_brightness = current_brightness + 10
            set_brightness(new_brightness)
        elif current_brightness > 90:
            new_brightness = current_brightness + (10 - (current_brightness % 10))
            set_brightness(new_brightness)
    elif "decrease" in speech:
        if current_brightness >= 10:
            new_brightness = current_brightness - 10
            set_brightness(new_brightness)
        elif current_brightness < 10:
            new_brightness = current_brightness - (current_brightness % 10)
            set_brightness(new_brightness)
    elif "set" in speech or "what":
        digits = ''.join(filter(str.isdigit, speech))
        if "zero" in speech:
            set_brightness(0)
            new_brightness = 0 
        if digits:
            new_brightness = int(digits)
            set_brightness(new_brightness)
    speak(f"Current screen brightness is: {current_brightness}")

    
'''Close apps'''

def close_apps(speech):
    apps_to_close = ["Messenger", "YouTube", "iTunes", "Skype", "Notepad", "Spotify", "Trello", "Chrome", "chatgpt3.5", "Preview" , "whatsapp" , "fiverr" ]
     
    # Join the elements of the list into a string
    speech = ' '.join(speech)
    if "fibre"  in speech :
        speech= "fiverr"
    if "chat gpt" in speech or "ai" in speech:
        speech= "chatgpt3.5"
    for app_name in apps_to_close:
        if app_name.lower() in speech.lower():
            for window in gw.getWindowsWithTitle(app_name):
                window.close()

'''Open Window apps'''
def open_apps(speech):
    speech = ' '.join(speech)
    app_dict = {
        "vlc": r"vlc.exe",
        "evernote": r"Evernote.exe",
        "spotify": r"Spotify.exe",
        "text": r"notepad.exe",
        "messenger": r"C:\Users\sa776\OneDrive\Desktop\Messenger.lnk",
        "trello": r"Trello.exe",
        "feedly": r"feedly.exe",
        "whatsapp": r"C:\Users\sa776\OneDrive\Desktop\WhatsApp.lnk",
        "skype": r"Skype.exe",  
        "fantastical": r"Fantastical.exe", 
        "facebook": "https://www.facebook.com",
        "reddit": "https://www.reddit.com",
        "live score": "https://www.livescore.com",
        "livescore": "https://www.livescore.com",
        "gmail": "https://www.gmail.com",
        "ai": "https://chat.openai.com",
        "youtube": "https://www.youtube.com/",
        "fibre": "https://www.fiverr.com/users/mazgamerz/manage_gigs",
        "f i v e r r": "https://www.fiverr.com/users/mazgamerz/manage_gigs",

    }

    for app_name, app_action in app_dict.items():
        if app_name.lower() in speech.lower():
            if app_action.startswith("http"):
                webbrowser.open(app_action)
            else:
                subprocess.run(["start", app_action], shell=True)


'''Weather API data'''
def sunrise(data):
    # sunrise time
    print("Sunrise: " + datetime.datetime.fromtimestamp(data['sunrise_time']).strftime('%B %d %H:%M'))
    speak("Sunrise will be at " + datetime.datetime.fromtimestamp(data['sunrise_time']).strftime('%I:%M %p'))

def sunset(data):
    # sunset time
    print("Sunset: " + datetime.datetime.fromtimestamp(data['sunset_time']).strftime('%B %d %H:%M'))
    speak("Sunset will be at " + datetime.datetime.fromtimestamp(data['sunset_time']).strftime('%I:%M %p'))

def weather(speech, data, temp):
    # includes today, tomorrow and forecast
    weather_status = data['detailed_status'].strip("''")

    if "weather" and "today" in speech:
        # Today's weather
        speak("Today's weather: " + weather_status)
        speak("Temperature will average at " + str(round(temp['temp'])) + " Celcius")

    elif "weather" and "forecast" in speech:
        # Get Forecast for the next week
        forecast_week = owm.daily_forecast(f"Bahawalpur, Pakistan",'daily', limit=7)
        f = forecast_week.get_forecast()
        print("\nForecast for the next 7 days: ")
        for weather in islice(f, 1, None):
            unix_time = weather.get_reference_time('unix')
            print("Date: " + datetime.datetime.fromtimestamp(unix_time).strftime('%B-%d') +
                  "   Weather: " + weather.get_detailed_status())

    elif "weather" and "tomorrow" in speech:
        # Tomorrow's weather
        forecast_week = owm.daily_forecast(f"Bahawalpur, Pakistan", limit=2)
        f = forecast_week.get_forecast()
        print("\nTomorrow's Weather: ")
        for weather in f:
            unix_time = weather.get_reference_time('unix')
            tomorrow_weather = (datetime.datetime.fromtimestamp(unix_time).strftime('%B-%d') +
                                " " + weather.get_detailed_status())
        speak(tomorrow_weather)

def temperature(temp):
    #TODO - add temp for today and tomorrow
    # Temperature status
    speak("Temperature will average at " + str(round(temp['temp'])) + " Celcius")
    speak("Max Temperature will be " + str(round(temp['temp_max'])) + " Celcius")



def choose_weather(speech):
    
    # Get weather observation for Bahawalpur, Pakistan
    observation = owm.weather_manager().weather_at_place("Bahawalpur, Pakistan")
    w = observation.weather  # Use 'weather' attribute to get Weather object

    # Convert weather data to dictionary
    data = w.to_dict()

    # Get temperature data in Celsius
    temperature_data = w.temperature(unit='celsius')  # Use 'temperature' attribute

    # pick the right method
    if "weather" in speech:
        weather(speech, data, temperature_data)
    elif "temperature" in speech:
        temperature(temperature_data)
    elif "sunrise" in speech:
        sunrise(data)
    elif "sunset" in speech:
        sunset(data)

'''Provides dates information'''
def date(speech):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dayNumber = datetime.datetime.today().weekday()
    if "time" in speech and "today" in speech:
        speak("now time is ")
        speak(datetime.datetime.now().strftime("%I:%M %p"))
    if "today" in speech or "date"in speech :
        speak(f"Today is {days[dayNumber]} and date is")        
        speak(datetime.datetime.now().strftime("%B %d"))
    if "tomorrow" in speech:
        dayNumber = dayNumber + 1
        if dayNumber == 7:
            speak(f"Tomorrow will be Monday and date willbe") 
            speak((datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%B %d"))
        else:
            speak(f"Tomorrow will be {days[dayNumber]} and date will be") 
            speak((datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%B %d"))

# Run these methods
if __name__ == "__main__":
    run = login()
