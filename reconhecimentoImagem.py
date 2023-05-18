import cv2
import face_recognition as fr

imgBillie = fr.load_image_file('Billie.jpg')
imgBillieTeste = fr.load_image_file('billie2.jpg')

#Transforma cor da imagem em RGB
imgBillie = cv2.cvtColor(imgBillie, cv2.COLOR_BGR2RGB)
imgBillieTeste = cv2.cvtColor(imgBillieTeste, cv2.COLOR_BGR2RGB)

#Localiza a face na imagem (coordenadas)
faceLocBillie = fr.face_locations(imgBillie)[0]
cv2.rectangle(imgBillie,(faceLocBillie[3], faceLocBillie[0]), (faceLocBillie[1], faceLocBillie[2]), (0, 255, 0),2)
#print(faceLocBillie)

#Extrai as 128 medidas do rosto da imagem
encodeBillie = fr.face_encodings(imgBillie)[0]
encondeBillieTeste= fr.face_encodings(imgBillieTeste)[0]

comparacao = fr.compare_faces([encodeBillie], encondeBillieTeste)
print(comparacao)

#print(encodeBillie)


cv2.imshow('Billie', imgBillie)
cv2.imshow('Billie Teste', imgBillieTeste)
cv2.waitKey(0)
