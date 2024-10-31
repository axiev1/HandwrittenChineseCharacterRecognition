# import libraries
import serial
import numpy as np
from tensorflow import keras
import cv2
import io
from matplotlib import pyplot as plt
import csv
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import pinyin as pinyin


def getImage():
    # open serial port to communicate with arduino
    ser = serial.Serial('COM3', 500000)

    # read success messages from arduino
    print(ser.readline())
    print(ser.readline())

    # tell arduino to take an image and send it back via serial
    ser.write(b"1")

    # read other success messages from arduino
    print(ser.readline())
    print(ser.readline())
    print(ser.readline())

    # read number of bytes the arduino is sending
    numBytes = ser.readline()
    print(int(numBytes))

    # read bytes into a byte array
    byteImg = ser.read(int(numBytes))

    # convert byte array into a pillow image
    PILimage = Image.open(io.BytesIO(byteImg)).convert("RGB")

    return PILimage


def getCharacter(PILimage):
    # convert pillow image into opencv, grayscale image, binarize (threshold it), and find contours of image
    opencvImage = cv2.cvtColor(np.array(PILimage), cv2.COLOR_RGB2BGR)
    plt.imshow(opencvImage, cmap="gray")
    plt.show()

    grey = cv2.cvtColor(opencvImage.copy(), cv2.COLOR_BGR2GRAY)
    plt.imshow(grey, cmap="gray")
    plt.show()

    ret, threshed = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    plt.imshow(threshed, cmap="gray")
    plt.show()

    contours, _ = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find bounding box of mandarin character
    coords = [10000, 10000, 0, 0]
    for c in contours:
        # filter out very small contours
        if cv2.contourArea(c) > 10:
            # get bounding box coordinates of contours
            [x, y, w, h] = cv2.boundingRect(c)
            # get bounding box containing all countors
            coords = [min(x, coords[0]), min(y, coords[1]), max(x + w, coords[2]), max(y + h, coords[3])]
            # draw rectangle around contour
            cv2.rectangle(opencvImage, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

    # crop character out of photo
    character = threshed[coords[1]: coords[3], coords[0]: coords[2]]
    # save pillow image

    charcopy = character.copy()

    # resize character to size 48 by 48
    character = cv2.resize(character, (48, 48))

    # invert colors
    character = cv2.bitwise_not(character)

    # draw box around character in original image and display
    cv2.rectangle(opencvImage, (coords[0], coords[1]), (coords[2], coords[3]), color=(0, 255, 0), thickness=2)
    plt.imshow(opencvImage, cmap="gray")
    plt.show()

    # display character
    plt.imshow(character, cmap="gray")
    plt.show()

    return character, charcopy


def getClassNames():
    # read class names from file
    with open('class_names.txt') as f:
        class_names = f.read().splitlines()
    return class_names


def predict(character):
    # loading keras model
    path = r"C:\Users\alexx\$ML_PATH\chinese\hccrmodel.keras"
    model = keras.models.load_model(path)

    # predict using keras model and output result
    prediction = model.predict(character.reshape(1, 48, 48))
    return prediction


def resizeImage(image):
    image = cv2.bitwise_not(image)
    height, width = image.shape[0], image.shape[1]
    scalar = 250/max(width, height)
    image = cv2.resize(image, (int(width*scalar), int(height*scalar)))
    height, width = image.shape[0], image.shape[1]
    right = (250-width)//2
    left = (250-width)//2
    top = (250-height)//2
    bottom = (250-height)//2
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    color_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pilChar = Image.fromarray(color_converted)
    return pilChar


def displayDict(top5, image):
    unihan = {}
    chars = []
    balls = r"unihan.csv"
    with open(balls, encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            unihan[row['char']] = row['kDefinition']

    for i in range(5):
        if top5[i][0] in unihan:
            chars.append([top5[i][0], unihan[top5[i][0]], top5[i][1]])
        else:
            chars.append([top5[i][0], "special character", top5[i][1]])

    # Aspects of the background:
    window = tk.Tk()
    window.configure(background="white")
    window.title("Mandarin Translation")

    # Displaying text/data/images on screen
    char_label = tk.Label(window, text=top5[0][0], font=("Arial", 40), bg="white")
    char_label.grid(row=0, column=21, rowspan=3, columnspan=3)

    pinyin_label = tk.Label(window, text=pinyin.get(chars[0][0]), font=("Arial", 15), bg="white")
    pinyin_label.grid(row=3, column=21, columnspan=3)

    definition = tk.Label(window, text="Definition: " + chars[0][1], font=("Arial", 10, "bold"), bg="white")
    definition.grid(row=4, column=21, columnspan=3)

    img = ImageTk.PhotoImage(image)
    l1 = Label(window, image=img, bg="black")
    l1.grid(row=0, column=0, rowspan=20, columnspan=20, padx=10, pady=10)  # do not mess with span for the love of god

    top_label = tk.Label(window, text="Top 5 predictions:", bg="white", anchor='w')
    top_label.grid(row=5, column=21, columnspan=3, sticky=W)
    rowPlacement = 5
    for prediction, definition, confidence in chars:
        rowPlacement += 1
        if prediction in unihan:
            pred_label = tk.Label(text=prediction + " (" + pinyin.get(prediction) + "): ", bg="white", anchor='w')
        else:
            pred_label = tk.Label(text=prediction + ": ", bg="white", anchor='w')
        confidence_label = tk.Label(window, text="{:0.2f}%".format(confidence * 100), bg="white", anchor='w')
        pred_label.grid(row=rowPlacement, column=21, sticky=W)
        confidence_label.grid(row=rowPlacement, column=22, sticky=W)

    window.mainloop()


PILimage = getImage()
character, charcopy = getCharacter(PILimage)

class_names = getClassNames()

prediction = predict(character)
idx = prediction[0].argsort()[-5:][::-1]

top5 = [(chr(int(class_names[idx[i]])), prediction[0][idx[i]]) for i in range(5)]

pilChar = resizeImage(charcopy)
displayDict(top5, pilChar)

print("Final Output: {}".format(chr(int(class_names[np.argmax(prediction)]))))
