import cv2
import os


video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
count = 0
nameID = str(input("Введите свое имя: ")).lower()
path = "image/" + nameID
isExist = os.path.exists(path)


if isExist:
    print("Папка с таким именем уже существует")
    nameID = str(input("Введите другое имя: "))
else:
    os.makedirs(path)


while True:
    ret, frame = video.read()
    faces = facedetect.detectMultiScale(frame, 1.3, 5)
    for x, y, w, h in faces:
        count = count + 1
        name = "./image/" + nameID + f"/{nameID}." + str(count) + ".png"
        print(name)
        print("Создание изображения " + name)
        cv2.imwrite(name, frame[y : y + h, x : x + w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.imshow("WindowFrame", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
    if count > 10000:
        break


video.release()
cv2.destroyAllWindows()
