from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
import imutils
import numpy as np
import cv2
import os

#YSA= Yapay Sinir Ağı

def algila(goruntu, sinirag, model):

	(y, g) = goruntu.shape[:2] #goruntun yukseklik genislik al
	blob = cv2.dnn.blobFromImage(goruntu, 1, (299, 299))      #resim, scale , size en iyi sonuç için(224,227,299) kullan
	# goruntuyu bloblayıp ysa gönder
	sinirag.setInput(blob)
	algilananyuz = sinirag.forward()

	#listeleri olustur.
	yuzler = []
	konumu = []
	tahminedilen = []

	# algilanan yuz dongusu
	for i in range(0, algilananyuz.shape[2]):
		dogruluk = algilananyuz[0, 0, i, 2]

		if dogruluk > 0.7: # yüzde 70 altı durumları gösterme maskeli maskesiz ddurumu 
			#yuz nesnesinin x,y kordinatlarını hesapla daha sonra değişkenlere at
			box = algilananyuz[0, 0, i, 3:7] * np.array([g, y, g, y])
			(x1, y1, x2, y2) = box.astype("int")

			# goruntunun kordinatları
			(x1, y1) = (max(0, x1), max(0, y1))
			(x2, y2) = (min(g - 1, x2), min(y - 1, y2))

			#goruntuyu resize et, goruntuyu rgb ye çevir
			yuz = goruntu[y1:y2, x1:x2]
			yuz = cv2.cvtColor(yuz, cv2.COLOR_BGR2RGB)#rgb cevir #önemli yoksa çok saçmalıyor.
			yuz = cv2.resize(yuz, (200, 200)) #goruntuyu resize et
			yuz = img_to_array(yuz) #goruntuyu arraye çevir
			yuz = preprocess_input(yuz)
			#yuzler listesine yuzu ekle
			#yuzun konumu bilgilerini konumu listesine ekle
			yuzler.append(yuz)
			konumu.append((x1, y1, x2, y2))

	if len(yuzler) > 0: #yuzler listesi uzunluğu >0 ise
		yuzler = np.array(yuzler, dtype="float32") #yuzler listesini np arrayine çevir
		tahminedilen = model.predict(yuzler, batch_size=50)#hizli tahmin için batch size arttırabilirsin
	return (konumu, tahminedilen)


model = load_model("maske_algilama.model")#maske modeli yukle
sinirag = cv2.dnn.readNet(r"yuz_tespit\proto.prototxt", r"yuz_tespit\agirlik.caffemodel") #ysa yukle



kamera=cv2.VideoCapture(0)#Kamera AÇ   video kaynağı 0 dahili 1 harici vs. 
while True: #Kamera kapatılana kadar döngü
	ret,goruntu=kamera.read()
	(konumu, tahminedilen) = algila(goruntu, sinirag, model) #algila fonk cagir yuzu algilat. konumu ve tahminedilen'i al


	for (cerceve, tahmin) in zip(konumu, tahminedilen):
		(x1, y1, x2, y2) = cerceve
		(maskeli, maskesiz) = tahmin #iki tahminide al maskeli maskesiz değişkenlerine at

		if(maskeli > maskesiz): #eger maskelinin tahmin orani yuksekse
			aciklama = "Maskeli"
			renk = (0, 255, 0)#yesil
		else:
			aciklama = "Maskesiz" #eger maskesizin tahmin orani yuksekse
			renk = (0, 0, 255)#kirmizi
		
		aciklama = "{}: %{}".format(aciklama, int(max(maskeli, maskesiz)*100)) #ekranda gözükecek cerceve ve texti ##yüzdeli yazma

		cv2.rectangle(goruntu, (x1-20, y1-50), (x2+20, y2+20), renk, 3)#ekranda gozukecek dikdortgen rengi ve boyutu
		cv2.putText(goruntu, aciklama, (x1-20, y1 - 60),cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 3)#ekranda gozukecek aciklama konumu  yazi tipi size rengi boyutu
		
	goruntu=imutils.resize(goruntu, width=1000) #ekranda gozukecek kamera boyutu buyutuyorum.
	cv2.imshow("Webcam. CIKMAK icin 'q' tusuna basiniz",goruntu) #goruntuyu göster

	if ((cv2.waitKey(41) & 0xFF) == ord("q")): #41 ms de bi görüntü al yani 1 saniyede 24 kare , waitletiyorum burda
		break  #q ye basılırsa boz

kamera.release()
cv2.destroyAllWindows()