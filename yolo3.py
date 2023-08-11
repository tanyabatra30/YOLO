import cv2
import numpy as np
import matplotlib.pyplot as plt

# upload file to environment
yolo = cv2.dnn.readNet("./yolov3-tiny.weights","./yolov3-tiny.cfg") # pre trained model ?

# import classes
classes = []
with open("./coco.names",'r') as f :
  classes = f.read().splitlines()

# loading image
img = cv2.imread("./cat_on_table.jpeg")

#creating a preprocessed image called blob of size 320x320 
blob = cv2.dnn.blobFromImage(img,1/255,(320,320),(0,0,0),swapRB = True ,crop = False)

# printing image
i = blob[0].reshape(320,320,3) 
plt.imshow(i)

# yolo setup and pushing blob
yolo.setInput(blob)
output_layers_name = yolo.getUnconnectedOutLayersNames()
layeroutput = yolo.forward(output_layers_name)

# cat_on_table dimension
width = 250
height = 200

# capture bounding box list
boxes = []
confidences = []
class_ids = []

for output in layeroutput :
  for detection in output :
    score = detection[5:]
    class_id = np.argmax(score)
    confidence = score[class_id]
    if confidence > 0.6:
      center_x = int(detection[0]*width)
      center_y = int(detection[1]*height)
      w = int(detection[2]*width)
      h = int(detection[3]*height)
      x=int(center_x- w/2)
      y=int(center_y- h/2)

      boxes.append([x,y,w,h])
      conf = float(confidence)
      confidences.append(conf)
      class_ids.append(class_id)
      
      len(boxes)

      indexes = cv2.dnn.NMSBoxes(boxes, confidences,0.5,0.4)

      font = cv2.FONT_HERSHEY_PLAIN
      colors = np.random.uniform(0,255,size=(len(boxes),3))

      # displaying image with labels and confidence level 
      for i in indexes.flatten():
        x,y,w,h = boxes[i]

        label = str(classes[class_ids[i]])
        confi = str(round(confidences[i],2))
        color = colors[i]

        cv2.rectangle(img , (x,y),(x+w , y+h), color , 2)
        cv2.putText(img, label + " "+confi, (x,y+20),font,2,(255,255,255),2)

        plt.imshow(img)
        
# save image
cv2.imwrite("./img.jpg",img)
