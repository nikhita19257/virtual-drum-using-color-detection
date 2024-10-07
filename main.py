import cv2
import numpy as np
from pygame import mixer
import time

# Initialize the webcam and audio mixer
webcam = cv2.VideoCapture(0)

ret, frame = webcam.read()
if not ret:
    print("Unable to read video")
    exit()

mixer.init()

# Drum class
class Drum:
    def __init__(self, pos, drum_type):  # Corrected constructor name
        self.position = pos
        self.type = drum_type
        self.sound = mixer.Sound(self.type + "_sound.mp3")
        self.image = cv2.resize(cv2.imread(self.type + ".png"), (self.position[2], self.position[3]))
        self.lastTime = time.time() - 1
        self.hit_count = 0
        self.hit_interval = 0
        self.hit_speed = 0

    def playSound(self):
        currentTime = time.time()
        interval = currentTime - self.lastTime
        if interval < 0.5:
            return
        self.sound.play()
        self.lastTime = currentTime

        # Update hit count and speed
        self.hit_count += 1
        self.hit_interval = interval
        self.hit_speed = 1 / self.hit_interval if self.hit_interval > 0 else 0

    def addImage(self, frame):
        o = frame[self.position[1]:self.position[1] + self.position[3],
                  self.position[0]:self.position[0] + self.position[2]]

        frame[self.position[1]:self.position[1] + self.position[3],
              self.position[0]:self.position[0] + self.position[2]] = cv2.addWeighted(self.image, 0.6, o, 0.4, 0.0)

        # Display the hit speed and beat count on the drum image
        cv2.putText(frame, f"Speed: {self.hit_speed:.2f} Hz", (self.position[0], self.position[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Beats: {self.hit_count}", (self.position[0], self.position[1] + self.position[3] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def checkHit(self, pos):
        if (self.position[0] <= pos[0] <= self.position[0] + self.position[2]) and (
                self.position[1] <= pos[1] <= self.position[1] + self.position[3]):
            self.playSound()

# Drums
snare_drum = Drum((100, 200, 100, 100), "snare_drum")
bass_drum = Drum((200, 300, 100, 100), "bass_drum")
hi_hat = Drum((500, 200, 100, 100), "hi_hat")
tom_drum = Drum((400, 300, 100, 100), "tom_drum")

drums_list = [snare_drum, bass_drum, hi_hat, tom_drum]

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Unable to read video")
        break
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for blue color and create mask
    blue_lower = np.array([90, 50, 70])
    blue_upper = np.array([128, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Define range for green color and create mask
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Combine the masks
    combined_mask = cv2.bitwise_or(blue_mask, green_mask)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_ERODE, np.ones((5, 5), np.uint8))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))
    cv2.imshow("Mask", combined_mask)

    # Find contours and check for hits
    contours, _ = cv2.findContours(combined_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

        for drum in drums_list:
            drum.checkHit((int(x), int(y)))

    # Add drum images to the frame
    for drum in drums_list:
        frame = drum.addImage(frame)

    cv2.imshow("Magic Drum", frame)

    if cv2.waitKey(1) == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()

