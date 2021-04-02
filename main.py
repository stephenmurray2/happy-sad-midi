import cv2
import sys
import rtmidi
import numpy as np
import emoji
from fer import FER

from rtmidi.midiconstants import NOTE_OFF, NOTE_ON, ALL_NOTES_OFF

current_emotion = "happy"
happy_colour = (182, 237, 149)
sad_colour = (255, 108, 59)
surprise_colour = (112, 71, 245)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)

midiout = rtmidi.MidiOut()
available_ports = midiout.get_ports()

if midiout.get_ports(): 
    midiout.open_port(0)
else:
    midiout.open_virtual_port("realtime_annotate.py")

with midiout:
    # define the midi events
    D_2_on = [NOTE_ON, 50, 112] 
    G_2_on = [NOTE_ON, 55, 112] 
    A_2_on = [NOTE_ON, 57, 112] 
    C_sharp_on = [NOTE_ON, 61, 112] 
    D_on = [NOTE_ON, 62, 112] 
    E_on = [NOTE_ON, 64, 112]
    F_sharp_on = [NOTE_ON, 66, 112]
    G_on = [NOTE_ON, 67, 112]
    A_on = [NOTE_ON, 69, 112]
    B_on = [NOTE_ON, 71, 112]

    # create the CNN detector
    detector = FER(mtcnn=True)

    colour = happy_colour
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(200, 200),
            flags=cv2.CASCADE_SCALE_IMAGE #updated identifier name in OpenCV 3 I think
        )

        print(faces)
        # Draw a rectangle around the faces
        if (len(faces) > 0):
            for (x, y, w, h) in [faces[0]]:
                cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 3)
                
                # draw a little face, because why not?

                # face outline
                cv2.circle(frame, (x+w+28,y+20), 20, colour, 3)
                # two eyes
                cv2.circle(frame, (x+w+20,y+12), 1, colour, 3)
                cv2.circle(frame, (x+w+36,y+12), 1, colour, 3)
                
                # if you're happy and you know it draw a smile
                if (current_emotion == "happy"):
                    cv2.ellipse(frame, (x+w+28,y+20), (12, 12), 0, 45, 135, colour, 3)
                elif (current_emotion == "sad"):
                    cv2.ellipse(frame, (x+w+28,y+40), (16, 16), 0, 225, 315, colour, 3)
                elif (current_emotion == "surprise"):
                    cv2.circle(frame, (x+w+28,y+30), 5, colour, 3)
                
                roi_gray = gray[y:y + h, x:x + w]
                roi_colour = frame[y:y + h, x:x + w]
                a = np.asarray(roi_colour[:,:])
                arr = detector.detect_emotions(a)
                if (len(arr) > 0):
                    happy = arr[0].get('emotions').get('happy')
                    sad = arr[0].get('emotions').get('sad')
                    surprise = arr[0].get('emotions').get('surprise')
                    if (happy > sad and happy > surprise and current_emotion != 'happy'):
                        midiout.send_message(D_2_on)
                        midiout.send_message(D_on)
                        midiout.send_message(F_sharp_on)
                        midiout.send_message(A_on)
                        current_emotion = "happy"
                        colour = happy_colour
                    elif (sad > happy and sad > surprise and current_emotion != "sad"):
                        midiout.send_message(A_2_on)
                        midiout.send_message(C_sharp_on)
                        midiout.send_message(E_on)
                        midiout.send_message(A_on)
                        current_emotion = "sad"
                        colour = sad_colour
                    elif (surprise > happy and surprise > sad and current_emotion != "surprise"):
                        midiout.send_message(G_2_on)
                        midiout.send_message(D_on)
                        midiout.send_message(G_on)
                        midiout.send_message(B_on)
                        current_emotion = "surprise"
                        colour = surprise_colour

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

del midiout

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()