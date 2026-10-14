import os
os.environ["KIVY_NO_CONSOLELOG"] = "1"

import cv2
import threading
import pyttsx3
import speech_recognition as sr

from ultralytics import YOLO

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import (
    Color,
    RoundedRectangle,
    Line
)

from kivy.graphics.texture import Texture

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label


# LOAD AI MODE

model = YOLO("yolov8n.pt")


# VOICE ENGINE


engine = pyttsx3.init()

engine.setProperty(
    'rate',
    170
)


# NAVI UI


class NaviUI(BoxLayout):

    def __init__(self, **kwargs):

        super().__init__(
            orientation='vertical',
            padding=15,
            spacing=15,
            **kwargs
        )

        # VARIABLES

        self.camera_running = False
        self.detect_objects = True
        self.cap = None
        self.last_object = ""
        

        
        # BACKGROUND
        

        with self.canvas.before:

            Color(
                0.05,
                0.05,
                0.09,
                1
            )

            self.bg = RoundedRectangle(
                pos=self.pos,
                size=self.size
            )

        self.bind(
            pos=self.update_bg,
            size=self.update_bg
        )

    
        # TITLE
    

        self.title = Label(
            text="[b]NAVI AI ASSISTANT[/b]",
            markup=True,
            font_size=36,
            color=(0, 1, 1, 1),
            size_hint=(1, 0.12)
        )

    
        # CAMERA BOX
    

        self.camera_box = BoxLayout(
            size_hint=(1, 0.65),
            padding=5
        )

        with self.camera_box.canvas.before:

            Color(
                0,
                0.7,
                1,
                1
            )

            self.camera_border = Line(
                rounded_rectangle=(
                    0,
                    0,
                    100,
                    100,
                    20
                ),
                width=2
            )

        self.camera_box.bind(
            pos=self.update_camera_border,
            size=self.update_camera_border
        )

        self.image = Image()

        self.camera_box.add_widget(
            self.image
        )

        
        # STATUS
        

        self.status = Label(
            text="Press START NAVI",
            font_size=20,
            color=(0, 1, 0, 1),
            size_hint=(1, 0.08)
        )

        
        # BUTTONS
        

        btn_layout = BoxLayout(
            size_hint=(1, 0.15),
            spacing=15
        )

        self.start_btn = Button(
            text="START",
            font_size=22,
            bold=True,
            background_normal='',
            background_color=(
                0,
                0.8,
                0,
                1
            )
        )

        self.stop_btn = Button(
            text="STOP",
            font_size=22,
            bold=True,
            background_normal='',
            background_color=(
                1,
                0,
                0,
                1
            )
        )

        self.start_btn.bind(
            on_press=self.start_camera
        )

        self.stop_btn.bind(
            on_press=self.stop_camera
        )

        btn_layout.add_widget(
            self.start_btn
        )

        btn_layout.add_widget(
            self.stop_btn
        )

        
        # ADD WIDGETS
    

        self.add_widget(self.title)
        self.add_widget(self.camera_box)
        self.add_widget(self.status)
        self.add_widget(btn_layout)

    
        # START VOICE LISTENER
        

        threading.Thread(
            target=self.listen_commands,
            daemon=True
        ).start()

        # Welcome Voice

        threading.Thread(
            target=self.welcome_voice,
            daemon=True
        ).start()

    
    # UPDATE BACKGROUND


    def update_bg(self, *args):

        self.bg.pos = self.pos
        self.bg.size = self.size

    # UPDATE CAMERA BORDER


    def update_camera_border(
        self,
        *args
    ):

        self.camera_border.rounded_rectangle = (
            self.camera_box.x,
            self.camera_box.y,
            self.camera_box.width,
            self.camera_box.height,
            20
        )
    # SPEAK
    

    def speak(self, text):

        self.status.text = text

        engine.say(text)
        engine.runAndWait()

    
    # WELCOME
    

    def welcome_voice(self):

        self.speak(
            "Hello, I am NAVI AI Assistant"
        )

    # START CAMERA
    

    def start_camera(
        self,
        instance=None
    ):

        if self.camera_running:
            return

        self.camera_running = True

        self.speak(
            "Opening camera"
        )

        self.cap = cv2.VideoCapture(0)

        Clock.schedule_interval(
            self.update,
            1.0 / 30.0
        )


    # STOP CAMERA


    def stop_camera(
        self,
        instance=None
    ):

        self.camera_running = False

        self.speak(
            "Closing camera"
        )

        if self.cap:

            self.cap.release()
            self.cap = None


    # CAMERA UPDATE


    def update(self, dt):

        if not self.camera_running:
            return

        ret, frame = self.cap.read()

        if not ret:
            return

        frame = cv2.resize(
            frame,
            (640, 480)
        )

        
        # OBJECT DETECTION


        if self.detect_objects:

            results = model(
                frame,
                imgsz=320,
                conf=0.5,
                verbose=False
            )

            annotated_frame = results[0].plot()

            for box in results[0].boxes:

                cls = int(box.cls[0])

                confidence = float(
                    box.conf[0]
                )

                if confidence > 0.65:

                    name = model.names[cls]

                    if name != self.last_object:

                        self.last_object = name

                        threading.Thread(
                            target=self.speak,
                            args=(
                                f"{name} detected",
                            ),
                            daemon=True
                        ).start()

        else:

            annotated_frame = frame

    
        # SHOW FRAME
    

        buffer = cv2.flip(
            annotated_frame,
            0
        ).tobytes()

        texture = Texture.create(
            size=(
                annotated_frame.shape[1],
                annotated_frame.shape[0]
            ),
            colorfmt='bgr'
        )

        texture.blit_buffer(
            buffer,
            colorfmt='bgr',
            bufferfmt='ubyte'
        )

        self.image.texture = texture


    # VOICE COMMANDS


    def listen_commands(self):

        recognizer = sr.Recognizer()

        while True:

            try:

                with sr.Microphone() as source:

                    self.status.text = (
                        "Listening..."
                    )

                    recognizer.adjust_for_ambient_noise(
                        source,
                        duration=1
                    )

                    audio = recognizer.listen(
                        source,
                        timeout=5,
                        phrase_time_limit=4
                    )

                command = recognizer.recognize_google(
                    audio
                ).lower()

                print(
                    "Command:",
                    command
                )

            
                # OPEN CAMERA
            

                if (

                    "open camera" in command or
                    "camera open" in command or
                    "camera chalu karo" in command or
                    "कैमरा खोलो" in command

                ):

                    self.start_camera()

            
                # CLOSE CAMERA
            

                elif (

                    "close camera" in command or
                    "camera band karo" in command or
                    "camera close" in command or
                    "कैमरा बंद करो" in command

                ):

                    self.stop_camera()

            
                # STOP DETECTION
                

                elif (

                    "stop detection" in command or
                    "navi stop" in command

                ):

                    self.detect_objects = False

                    self.speak(
                        "Detection paused"
                    )

    
                # START DETECTION
                

                elif (

                    "start detection" in command or
                    "navi start" in command

                ):

                    self.detect_objects = True

                    self.speak(
                        "Detection resumed"
                    )

            

                elif (

                    "close app" in command or
                    "app close" in command

                ):

                    self.speak(
                        "Closing application"
                    )

                    App.get_running_app().stop()

                    break

            except:

                pass


class NaviApp(App):

    def build(self):

        return NaviUI()


NaviApp().run()