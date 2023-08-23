import os, cv2
image_path = "images"

def save_faces(cascade, imgname):
    img = cv2.imread(os.path.join(image_path, imgname))
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i, face in enumerate(cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=14, minSize=(100,100))):
        x, y, w, h = face
        sub_face = img[y:y + h, x:x + w]
        cv2.imwrite(os.path.join("/mnt/c/Users/saura/Documents/Android/faces", "{}_{}.jpg".format(imgname[:-4], i)), sub_face)

if __name__ == '__main__':
    face_cascade = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(face_cascade)
    # Iterate through files
    for f in [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]:
        save_faces(cascade, f)