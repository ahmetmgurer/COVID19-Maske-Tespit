#ilgili paket kütüphaneler
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import os


INIT = 1e-4 # ilk ögrenme oranı. Default olarak 1e-3 fakat öğrenme çok inişli çıkışlı oluyor
EPOCH = 30 #epoch sayisi / verisetini kaç kez tekrar edileceği
BATCH = 100 #batch boyutu

KONUM = r"C:\Users\muco\Desktop\bitirmeproj\veriseti"
TUR = ["maskeli", "maskesiz"]


print("Fotoğraflar Yükleniyor..")

dizi = [] #veri listesi
etiket = [] #etiket listesi

for tip in TUR:
    yol = os.path.join(KONUM, tip)
    for resim in os.listdir(yol):
    	img_yol = os.path.join(yol, resim)
    	grntu = load_img(img_yol, target_size=(200, 200)) #resim boyutunu buyutunce islem cok uzun suruyor ve sonuclar hatali cikiyor #resmin yolu ve boyutu
    	grntu = img_to_array(grntu) #resim to dizi cevir
    	grntu = preprocess_input(grntu)

    	dizi.append(grntu) #listeye ekle
    	etiket.append(tip) #listeye ekle

# etiket kodlama
lbl = LabelBinarizer()
etiket = lbl.fit_transform(etiket)
etiket = to_categorical(etiket)

##numpy dizisi
dizi = np.array(dizi, dtype="float32") #veri, veritipi
etiket = np.array(etiket) #veri

(egtmX, testX, egtmY, testY) = train_test_split(dizi, etiket, test_size=0.25, stratify=etiket, random_state=50) #sklearn kutuphanesi splitle
#testsize 0-1 arası; verisetinin %25 sini test için kullanıyorum.
#trainsize 0-01 arası; verisetinin eğitim oranı için. %75 kalıyor
#randomstate; int alır. veri split edilmeden önce verilere uygulanan karıştırma / karma
#stratify sınıf etiketialır.

# eğitim görüntüsünü büyütmek için, hazır
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# MobileNetV2 ağı, FC katmanı baş düğüm ayarla
ModelTaban = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(200, 200,3))) #boyut ve katman ölçüleri 0 gri, 3 rgb

# temel modelden, modelin üst construck
#KERAS
ModelBas = ModelTaban.output
ModelBas = AveragePooling2D(pool_size=(7, 7))(ModelBas) #iki boyut için pencere uzunluğu
ModelBas = Flatten(name="flatten")(ModelBas)
ModelBas = Dense(128, activation="relu")(ModelBas) #çıktı uzayı boyutu, aktivasyon fonk tipi
ModelBas = Dropout(0.5)(ModelBas) #dropout katmanı, inputun Fraction adedi tf.keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)
ModelBas = Dense(2, activation="softmax")(ModelBas) #çıktı uzayı boyutu, aktivasyon fonk. 

#asil modeli egitecek. model taban alacak modelbas cikti vercek
model = Model(inputs=ModelTaban.input, outputs=ModelBas)

# temel model katmanları arası döngü. ilk eğitim sürerken güncelleme yapmaz.
for layer in ModelTaban.layers:
	layer.trainable = False #ilk layer eğitme

# model derleme
print("Model Derleniyor..")
opt = Adam(lr=INIT, decay=INIT / EPOCH) #lr=learning rate öğrenme hızı. 
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

#ağın başının eğitilmesi
print("YSA nöron eğitimi başlıyor.")
H = model.fit(
	aug.flow(egtmX, egtmY, batch_size=BATCH), #
	steps_per_epoch=len(egtmX) // BATCH, #
	validation_data=(testX, testY),  #veri doğruluyor test verisiyle karşılaştırma
	validation_steps=len(testX) // BATCH, #
	epochs=EPOCH)

# test verisiyle kesinlik tahmin ediyor..
print("Test setiyle ağ değerlendiriyor...")
thmnid = model.predict(testX, batch_size=BATCH)


# tahmin edilen olasılığa sahip etiket
thmnid = np.argmax(thmnid, axis=1)


print(classification_report(testY.argmax(axis=1), thmnid, target_names=lbl.classes_))

# model diske kaydediliyor.
print("Model kaydeydiliyor.")
model.save("maske_algilama.model", save_format="h5")

# egitim sonucu yazdırılıyor.
N = EPOCH
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="eğitim_kayıp")
plt.plot(np.arange(0, N), H.history["val_loss"], label="test_kayıp")
plt.plot(np.arange(0, N), H.history["acc"], label="eğitim_doğruluk")
plt.plot(np.arange(0, N), H.history["val_acc"], label="test_doğruluk")
plt.title("Loss, Accuracy Tablosu")
plt.xlabel("Epoch Sayısı(30)")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="center right")
plt.savefig("egitimsonucu.png")