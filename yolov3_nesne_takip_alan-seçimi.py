import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture("cars.mp4")

#Seçilen koordinatları saklamak için boş bir liste oluştur
seçilen_koordinatlar = []

#Fare olayları işlendiğinde çağrılacak işlev
def show_coordinates(event, x, y, flags, param):
    global seçilen_koordinatlar

    if event == cv2.EVENT_LBUTTONDOWN:
        seçilen_koordinatlar = [(x, y)]  #Başlangıç koordinatını sakla
    elif event == cv2.EVENT_LBUTTONUP:
        seçilen_koordinatlar.append((x, y))  #Bitiş koordinatını sakla
        # Seçilen alanı kırmızı çerçeveyle çiz
        cv2.rectangle(frame, seçilen_koordinatlar[0], seçilen_koordinatlar[1], (0, 0, 255), 2)
        cv2.imshow("Nesne algilama", frame)

#Pencere oluştur ve fare olayları işlemek için işlevi ayarla
cv2.namedWindow("Nesne algilama")
cv2.setMouseCallback("Nesne algilama", show_coordinates)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Çerçeveyi YOLO modeline uygun formata dönüştür
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    # Tespit sonuçlarını işle
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Seçilen alan içindeki nesneleri tespit et
                if seçilen_koordinatlar and len(seçilen_koordinatlar) == 2:
                    x1, y1 = seçilen_koordinatlar[0]
                    x2, y2 = seçilen_koordinatlar[1]
                    if x1 < x < x2 and y1 < y < y2:
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in indexes:
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = colors[class_ids[i]]
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Nesne algilama", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
