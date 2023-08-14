#!/usr/bin/env python
# coding: utf-8

# Secara umum, akan dilakukan pemodelan menggunakan Convolutional Neural Network (CNN) untuk melakukan klasifikasi multikelas berdasarkan data image 2 kelas (Halal vs Syubhat). Beberapa tujuan utama dari kode berikut adalah:
# 1. Memuat dan mempersiapkan dataset: Kode tersebut memuat dataset gambar yang terdiri dari beberapa kelas. Kemudian, dataset tersebut diproses dan disiapkan agar sesuai dengan kebutuhan model CNN, seperti mengubah gambar menjadi grayscale, mengubah ukuran gambar, dan melakukan normalisasi piksel.
# 2. Membangun arsitektur model CNN: Kode tersebut mendefinisikan arsitektur model CNN menggunakan library TensorFlow. Model ini terdiri dari beberapa layer seperti Convolutional Layer, Max Pooling Layer, Flatten Layer, dan Dense Layer. Arsitektur model CNN ini dipilih untuk dapat mengekstraksi fitur-fitur penting dari gambar dan melakukan klasifikasi multikelas.
# 3.Melatih model: Setelah membangun model CNN, kode tersebut melatih model dengan menggunakan data latih. Selama proses pelatihan, model secara iteratif menyesuaikan bobot dan biasnya untuk mengoptimalkan kinerja model dalam melakukan klasifikasi.
# 4.Evaluasi model: Setelah pelatihan selesai, kode tersebut mengevaluasi kinerja model menggunakan data uji. Evaluasi ini melibatkan perhitungan loss (kerugian) dan akurasi model dalam melakukan klasifikasi gambar.
# 5. Visualisasi hasil: Selain itu, kode tersebut juga menyediakan visualisasi hasil pelatihan, seperti kurva pembelajaran (learning curves) yang menunjukkan perubahan loss dan akurasi model seiring dengan jumlah epoch.
# 6. Menggunakan model untuk prediksi: Setelah pelatihan dan evaluasi selesai, kode tersebut menggunakan model yang telah dilatih untuk melakukan prediksi pada data uji. Hasil prediksi ini kemudian digunakan untuk membangun confusion matrix dan classification report, yang memberikan informasi tentang kinerja model dalam melakukan klasifikasi pada setiap kelas.

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import os
import pandas as pd
import glob
import random
import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from IPython.core.display import HTML,display
import cv2
import numpy as np
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.python.framework import random_seed
from PIL import Image
from IPython.display import display, Image as IPImage
from IPython.display import display, HTML
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
np.random.seed(32)
tf.random.set_seed(32)
random.seed(32)
random_seed.set_seed(32)
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Agar operasi TensorFlow yang deterministik


# In[3]:


halal_dir='/content/drive/MyDrive/Jago Data/INVDL300423-CNN Multiclass Classification/dataset/Halal'
syubhat_dir='/content/drive/MyDrive/Jago Data/INVDL300423-CNN Multiclass Classification/dataset/Syubhat'


# In[4]:



# Membuat list kosong untuk menyimpan informasi file dan label
data = []

# Meloopi setiap subfolder dan memuat informasi file
for dir in [halal_dir, syubhat_dir]:
  if dir==halal_dir:
    kelas='Halal'
  else:
    kelas='Syubhat'
  for label in sorted(os.listdir(dir)):
      label_dir = os.path.join(dir, label)
      if os.path.isdir(label_dir):
          for filename in os.listdir(label_dir):
              file_path = os.path.join(label_dir, filename)
              data.append({"filename": file_path, "label": label, "class": kelas})

# Membuat dataframe dari data
df = pd.DataFrame(data)


# In[5]:


df.shape


# In[6]:


df.sample(10)


# In[7]:


# Mengatur ukuran gambar
plt.figure(figsize=(12, 5)) 
# Menghitung jumlah data per kelas
class_counts = df['class'].value_counts().sort_index()
# Membuat visualisasi bar chart
plt.bar(class_counts.index, class_counts.values)
# Memutar label sumbu x
plt.xticks(rotation=90)
for i, v in enumerate(class_counts.values):
    plt.text(i, v, str(v), ha='center', va='bottom')
# Menambahkan label pada sumbu x dan y
plt.xlabel('Kelas')
plt.ylabel('Jumlah Data')
# Menambahkan judul grafik
plt.title('Jumlah Data per Kelas')
# Menampilkan grafik
plt.show()


# In[8]:


image_formats = ["png", "jpg", "jpeg"];

def show_images(image_files,name): 
    display(HTML('<H4 style="color:blue"> <b>Sampel Data {} </b></H5><hr>'.format(name)))
    fig = plt.figure(figsize=(18,18))
    fig.patch.set_facecolor('xkcd:white')
    for i in range(len(image_files)):
        plt.subplot(2,5,i+1)    
        img=cv2.imread(image_files[i])
        plt.imshow(img)
        plt.tight_layout()
        plt.axis('off')
        plt.title(image_files[i].split("/")[-1]+"\n"+"{}x{}".format(img.shape[0], img.shape[1])) 
    plt.show()

def list_files(dir):
    arr = []
    for root, dirs, files in os.walk(dir):
        for name in files[0:5]:
            if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".jpeg"):
                arr.append(os.path.join(root, name))
    return arr


# ### Sample Halal Image

# In[9]:


nama_subfolder_halal = [nama for nama in os.listdir(halal_dir) if os.path.isdir(os.path.join(halal_dir, nama))]

for i in nama_subfolder_halal:
  show_images(list_files(os.path.join(halal_dir,i)), i)


# ### Sample Syubhat Image

# In[10]:


nama_subfolder_syubhat = [nama for nama in os.listdir(syubhat_dir) if os.path.isdir(os.path.join(syubhat_dir, nama))]

for i in nama_subfolder_syubhat:
  show_images(list_files(os.path.join(syubhat_dir,i)), i)


# ### Preprocessing Gambar
# Bagian ini melakukan beberapa tahap preprocessing pada gambar-gambar. Pertama, gambar-gambar diubah menjadi grayscale menggunakan cv2.cvtColor() untuk mengurangi kompleksitas dan dimensi data. Kemudian, gambar-gambar diubah menjadi ukuran 30x90 piksel menggunakan cv2.resize(). Selanjutnya, pixel values gambar dinormalisasi menjadi rentang 0-1 dengan membagi setiap piksel dengan 255. Proses ini membantu dalam mempersiapkan data untuk pelatihan model.

# In[11]:


# Membaca data gambar dan label
data = []
labels = []
filenames=[]
for index, row in df.iterrows():
    filename = row['filename']
    label = row['class']
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Mengubah gambar menjadi grayscale
    img = cv2.resize(img, (90, 30))  # Mengubah ukuran gambar menjadi 30x90
    img = img / 255.0  # Normalisasi pixel values menjadi range 0-1
    data.append(img)
    labels.append(label)
    filenames.append(filename)


# In[12]:


def show_images_transformed(image_files,name): 
    display(HTML('<H4 style="color:blue"> <b>Sampel Data {} </b></H5><hr>'.format(name)))
    fig = plt.figure(figsize=(18,18))
    fig.patch.set_facecolor('xkcd:white')
    for i in range(len(image_files)):
        plt.subplot(2,5,i+1)    
        img=cv2.imread(image_files[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Mengubah gambar menjadi grayscale
        img = cv2.resize(img, (90, 30))  # Mengubah ukuran gambar menjadi 30x90
        img = img / 255.0  # Normalisasi pixel values menjadi range 0-1
        plt.imshow(img, cmap='gray')
        plt.tight_layout()
        plt.axis('off')
        plt.title(image_files[i].split("/")[-1]+"\n"+"{}x{}".format(img.shape[0], img.shape[1])) 
    plt.show()


# ### Sample Halal Image Transformed

# In[13]:


for i in nama_subfolder_halal:
  show_images_transformed(list_files(os.path.join(halal_dir,i)), i)


# ### Sample Syubhat Image Transformed

# In[14]:


for i in nama_subfolder_syubhat:
  show_images_transformed(list_files(os.path.join(syubhat_dir,i)), i)


# Code di bawah merupakan bagian dari proses preprocessing pada dataset sebelum melatih model. 
# Berikut adalah penjelasan detil untuk setiap bagian kode tersebut:
# 1. Konversi Label ke One-Hot Encoding:
# 
# `label_encoder = LabelEncoder()`: Membuat objek LabelEncoder dari library sklearn yang digunakan untuk mengkodekan label kelas menjadi bilangan bulat.
# `labels_encoded = label_encoder.fit_transform(labels)`: Menggunakan objek LabelEncoder untuk mengkodekan label kelas menjadi bilangan bulat.
# `labels_encoded = to_categorical(labels_encoded, num_classes=2)`: Mengubah bilangan bulat yang sudah dikodekan menjadi representasi one-hot encoding menggunakan fungsi `to_categorical()` dari library tensorflow. One-hot encoding mengubah setiap bilangan bulat menjadi vektor biner dengan panjang yang sama dengan jumlah kelas, di mana hanya indeks yang sesuai dengan label kelas yang memiliki nilai 1, dan indeks lainnya memiliki nilai 0.
# 
# 2. Konversi Data dan Label ke Array NumPy:
# 
# `data = np.array(data)`: Mengkonversi data menjadi array NumPy. data merupakan variabel yang berisi gambar-gambar yang sudah diubah menjadi grayscale dan diresize.
# `labels = np.array(labels)`: Mengkonversi labels menjadi array NumPy. labels merupakan variabel yang berisi label kelas untuk setiap gambar.
# 
# 3. Pembagian Data menjadi Set Pelatihan dan Pengujian:
# 
# `x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=32, stratify=labels_encoded)`: Menggunakan fungsi `train_test_split()` dari library sklearn untuk membagi data dan label ke dalam set pelatihan dan pengujian. data dan labels_encoded adalah input data dan label yang sudah dikonversi ke array NumPy. test_size=0.2 mengatur proporsi data yang akan digunakan sebagai data pengujian menjadi 20% dari total data. `random_state=32` digunakan untuk menghasilkan pembagian data yang konsisten setiap kali kode dijalankan. `stratify=labels_encoded` digunakan untuk memastikan bahwa pembagian data dilakukan secara proporsional berdasarkan label kelas.
# Variabel `x_train` akan berisi data gambar yang digunakan sebagai set pelatihan, `y_train` akan berisi label kelas yang sesuai dengan set pelatihan, `x_test` akan berisi data gambar yang digunakan sebagai set pengujian, dan `y_test` akan berisi label kelas yang sesuai dengan set pengujian.
# 
# 4. Pembagian Set Pengujian menjadi Validasi dan Pengujian:
# 
# `x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=32, stratify=y_test)`: Menggunakan fungsi `train_test_split()` lagi untuk membagi set pengujian menjadi set validasi dan set pengujian. `x_test` dan `y_test` adalah input data dan label yang sudah dibagi sebelumnya. `test_size=0.5` mengatur proporsi data yang akan digunakan sebagai set validasi menjadi 50% dari total data pengujian. `random_state=32` digunakan untuk menghasilkan pembagian data yang konsisten setiap melakukan running train_test_split.

# In[15]:


# Konversi label ke one-hot encoding
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_encoded = to_categorical(labels_encoded, num_classes=2)

# Convert data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=32, stratify=labels_encoded)

# Split test set into validation and test sets
x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=32, stratify=y_test)

# Print the shapes of the resulting datasets
print('Training set:', x_train.shape, y_train.shape)
print('Validation set:', x_valid.shape, y_valid.shape)
print('Test set:', x_test.shape, y_test.shape)


# In[16]:


# Membangun model CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(30, 90, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
print(model.summary())
plot_model(model, show_shapes=True)


# Selanjutnya kode berikut menggambarkan pembangunan dan pelatihan model Convolutional Neural Network (CNN) untuk melakukan klasifikasi pada data gambar. 
# Berikut adalah penjelasan langkah demi langkahnya:
# 
# - Pertama, kita membangun model CNN menggunakan objek Sequential dari TensorFlow. Model ini akan memiliki beberapa lapisan untuk mengekstraksi fitur dan melakukan klasifikasi.
# - Lapisan pertama yang ditambahkan ke model adalah Conv2D. Ini adalah lapisan konvolusi yang menggunakan 32 filter dengan ukuran kernel 3x3. Fungsi aktivasi yang digunakan adalah ReLU (Rectified Linear Unit). Lapisan ini juga menentukan input_shape yang sesuai dengan ukuran gambar yang akan digunakan (30x90 piksel dengan 1 saluran warna/grayscale).
# - Setelah lapisan konvolusi, kita menambahkan lapisan MaxPooling2D. Ini adalah lapisan pengurangan ukuran yang menggunakan jendela 2x2 untuk mengambil nilai maksimum dari setiap jendela.
# - Langkah-langkah 2 dan 3 diulang dengan tambahan lapisan konvolusi dan pooling. Lapisan konvolusi kedua menggunakan 64 filter dengan ukuran kernel yang sama, dan dilanjutkan dengan lapisan MaxPooling2D.
# - Setelah lapisan-lapisan konvolusi dan pooling, kita menambahkan lapisan Flatten. Ini mengubah matriks hasil pooling menjadi vektor yang dapat digunakan sebagai input untuk lapisan-lapisan berikutnya.
# - Lapisan Dense berikutnya memiliki 128 unit dengan fungsi aktivasi ReLU. Ini adalah lapisan terhubung penuh yang bertujuan untuk melakukan ekstraksi fitur yang lebih kompleks dari representasi vektor yang dihasilkan sebelumnya.
# - Untuk menghindari overfitting, kita menambahkan lapisan Dropout dengan tingkat dropout sebesar 0.5. Lapisan ini secara acak mematikan sebagian unit selama pelatihan untuk mencegah ketergantungan yang berlebihan pada set fitur tertentu.
# - Lapisan terakhir adalah lapisan Dense dengan 2 unit (sesuai dengan jumlah kelas yang akan diklasifikasikan) dan fungsi aktivasi softmax. Lapisan ini menghasilkan probabilitas untuk masing-masing kelas.
# - model.summary() digunakan untuk mencetak ringkasan dari arsitektur model, termasuk jumlah parameter yang akan dilatih.
# - Selanjutnya, model dikompilasi dengan menggunakan optimizer Adam dengan learning rate sebesar 1e-3. Fungsi loss yang digunakan adalah categorical_crossentropy karena ini adalah masalah klasifikasi multikelas. Metrics yang dipilih adalah categorical_accuracy untuk mengukur akurasi model.
# - Callbacks seperti EarlyStopping, ReduceLROnPlateau, dan ModelCheckpoint digunakan selama pelatihan model. EarlyStopping akan menghentikan pelatihan jika tidak ada peningkatan yang signifikan dalam loss. ReduceLROnPlateau akan mengurangi learning rate jika tidak ada peningkatan dalam validation loss. ModelCheckpoint akan menyimpan bobot model terbaik berdasarkan akurasi validasi.
# - Model dilatih dengan menggunakan data latih (x_train dan y_train) selama 50 epoch dengan batch size sebesar 32.
# - Pelatihan model dilakukan menggunakan metode fit() pada objek model. Parameter x_train dan y_train adalah data latih yang digunakan. Kita juga menyertakan x_valid dan y_valid sebagai data validasi untuk memantau performa model selama pelatihan.
# - Setelah pelatihan selesai, kita bisa menguji model menggunakan data uji (x_test dan y_test) dengan menggunakan metode evaluate(). Metode ini menghitung loss dan akurasi model pada data uji.

# In[17]:



# Compile model dengan optimizer Adam
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 12, verbose = 1)
reduce_lr =  tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=6, factor=0.1, verbose=2)
mcp = tf.keras.callbacks.ModelCheckpoint(filepath = 'cnn_model_weights.h5', monitor = 'val_categorical_accuracy', 
                      save_best_only = True, verbose = 1)

# Melatih model dengan data latih
model_history=model.fit(
        x_train,
        y_train,
        epochs=50,
        batch_size=32,
        steps_per_epoch=len(x_train) // 32,
        validation_data=(x_valid, y_valid),
        validation_steps=len(x_valid) // 32,
        callbacks=[early_stopping, reduce_lr, mcp]
    )


# Evaluasi model dengan data uji
loss, accuracy = model.evaluate(x_test, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)


# Selanjutnya code di bawah ini digunakan untuk menghasilkan grafik yang menunjukkan perkembangan loss (kehilangan) dan accuracy (akurasi) dari model selama proses pelatihan (training) dan validasi (validation). Fungsi plot_learning_curves digunakan untuk membuat grafik tersebut.
# 
# Penjelasan dari kode tersebut sebagai berikut:
# 
# - Fungsi plot_learning_curves memiliki dua parameter, yaitu history yang merupakan objek History yang berisi catatan pelatihan model, dan model_name yang merupakan nama model untuk memberikan nama pada file gambar hasil plot.
# - Dalam fungsi tersebut, metrik yang akan diplot adalah loss, precision, recall, dan accuracy. Variabel metrics menyimpan daftar metrik tersebut.
# - Kemudian, dilakukan ekstraksi nilai-nilai loss dan accuracy untuk pelatihan dan validasi dari objek history.
# - Gambar dengan ukuran 18x6 piksel dibuat menggunakan fig = plt.figure(figsize=(18, 6)).
# - Dua subplot ditambahkan ke dalam gambar menggunakan fig.add_subplot(1, 2, 1) dan fig.add_subplot(1, 2, 2). Subplot pertama digunakan untuk menggambarkan grafik training-validation accuracy, sedangkan subplot kedua digunakan untuk menggambarkan grafik training-validation loss.
# - Pada masing-masing subplot, grafik untuk training accuracy/loss dan validation accuracy/loss digambarkan menggunakan ax.plot().
# - Label dan judul pada masing-masing subplot ditambahkan menggunakan fungsi ax.set_title(), ax.set_xlabel(), dan ax.set_ylabel().
# - Ticks pada sumbu x dan y diatur menggunakan ax.set_xticks() dan ax.set_yticks().
# - Grafik yang dihasilkan disimpan sebagai file gambar dengan nama 'Learning Curves {}.png' menggunakan plt.savefig().
# - Lalu, grafik yang dihasilkan ditampilkan menggunakan plt.show().

# In[18]:


def plot_learning_curves(history, model_name):
    metrics =  ['loss', 'precision', 'recall','accuracy']
    train_acc=history.history['categorical_accuracy']
    val_acc=history.history['val_categorical_accuracy']
    train_loss=history.history['loss']
    val_loss=history.history['val_loss']
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(train_acc, '-o',label='Train Acc.')
    ax.plot(val_acc, '--<', color='orange', label='Validation Acc.')
    ax.set_title("Training-Validation Accuracy",size=12)
    ax.legend(loc='best',fontsize=12)
    ax.set_xlabel('Epoch', size=12)
    ax.set_ylabel('Accuracy', size=12)
    ax.set_xticks(np.arange(0, len(train_acc)+5, 5))
    ax.set_yticks(np.arange(0.1,1.05,0.1))
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(train_loss,'-o', label='Train Loss')
    ax.plot(val_loss,'--<', color='orange',  label='Validation Loss')
    ax.set_title("Train-Validation Loss",size=12)
    ax.legend(loc='best',fontsize=12)
    ax.set_xlabel('Epoch', size=12)
    ax.set_ylabel('Loss', size=12)
    ax.set_xticks(np.arange(0, len(train_acc)+5, 5))
    ax.set_yticks(np.arange(0,3.5,0.5))
    plt.savefig('Learning Curves {}.png'.format(model_name))
    plt.title(f'Learning Curves')
    plt.show()


# In[19]:


plot_learning_curves(model_history, "CNN")


# - Setelah model dilatih, kita dapat menggunakan model untuk melakukan prediksi pada data uji (x_test) dengan menggunakan metode predict(). Prediksi ini menghasilkan probabilitas kelas untuk setiap sampel data uji.
# - Selanjutnya, menggunakan fungsi argmax() dari NumPy, kita mengambil indeks kelas dengan probabilitas tertinggi untuk setiap sampel prediksi. Dengan demikian, kita mendapatkan label prediksi dalam bentuk bilangan bulat.
# - Menggunakan fungsi argmax() lagi, kita mengambil indeks kelas dengan probabilitas tertinggi untuk setiap sampel data uji (y_test). Dengan ini, kita mendapatkan label kelas sebenarnya dalam bentuk bilangan bulat.
# - Menggunakan objek label_encoder yang telah kita buat sebelumnya, kita mengubah label kelas sebenarnya dan label prediksi dari bilangan bulat ke label asli dalam bentuk string.
# - Dalam kode di bawah, terdapat juga definisi variabel labels yang berisi daftar nama kelas yang sesuai dengan indeks kelas yang digunakan dalam encoding.
# - Setelah itu, kita membuat confusion matrix menggunakan fungsi confusion_matrix() dari library Scikit-learn. Confusion matrix adalah tabel yang digunakan untuk mengevaluasi performa model dengan membandingkan label sebenarnya dan label prediksi.
# - Menggunakan library seaborn, kita menghasilkan visualisasi confusion matrix dalam bentuk heatmap. Di sini, setiap sel pada heatmap mewakili jumlah prediksi yang benar atau salah untuk setiap kombinasi label.
# - Selanjutnya, kita membuat classification report menggunakan fungsi classification_report() dari Scikit-learn. Classification report memberikan metrik evaluasi kinerja model seperti presisi (precision), recall, f1-score, dan support untuk setiap kelas. Hasil classification report di-print pada layar.

# In[20]:



# Melakukan prediksi menggunakan model
y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Mengubah label one-hot encoding menjadi label kelas
y_true_labels = np.argmax(y_test, axis=1)

decoded_ytest= label_encoder.inverse_transform(y_true_labels)
decoded_ypred= label_encoder.inverse_transform(y_pred_labels)

tag=['Halal','Syubhat']
# Membuat confusion matrix
cm = confusion_matrix(decoded_ytest, decoded_ypred)
sns.set_theme(rc={'figure.figsize':(6,6)})
ax = sns.heatmap(cm,annot=True,cmap='Reds',fmt="g",cbar=False, xticklabels=tag,
                     yticklabels=tag)
ax.set_ylabel('True Labels')
ax.set_xlabel('Predicted Labels');
plt.title(f"Confusion Matrix On Test Data")
plt.show()

# Membuat classification report
cr = classification_report(y_true_labels, y_pred_labels,  target_names=tag, digits=4)
print("\nClassification Report:")
print(cr)


# Dalam output tersebut, kita dapat melihat metrik evaluasi performa model untuk setiap kelas dalam dataset. Selain itu, juga diberikan akurasi keseluruhan (accuracy) serta rata-rata dari metrik evaluasi untuk semua kelas (macro avg dan weighted avg). Terlihat bahwa akurasi yang diperoleh adaalah 98% pada data testing.

# ### Contoh Hasil Prediksi menggunakan Model CNN yang dibuat

# In[21]:


pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_colwidth', -1)

def get_thumbnail(path):
    i = Image.open(path)
    i.thumbnail((150, 150), Image.LANCZOS)
    return i

def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'


def sample_results(dire, nama_subfolder, kelas):
  actual_labels=[]
  filenames=[]
  predicted_labels=[]
  for i in nama_subfolder:
    sbfolders=list_files(os.path.join(dire,i))[0:1]
    data=[]
    for item in sbfolders:
      img = cv2.imread(item)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Mengubah gambar menjadi grayscale
      img = cv2.resize(img, (90, 30))  # Mengubah ukuran gambar menjadi 30x90
      img = img / 255.0  # Normalisasi pixel values menjadi range 0-1
      data.append(img)
    data=np.array(data)
    y_pred_new=model.predict(data)
    y_pred_new_label = np.argmax(y_pred_new, axis=1)
    y_pred_new_label = label_encoder.inverse_transform(y_pred_new_label)
    actual_labels.append(kelas)
    filenames.append(item)
    predicted_labels.append(y_pred_new_label[0])
  data = {'Filenames': filenames, 'Actual Labels': actual_labels, 'Predicted Labels': predicted_labels}
  df = pd.DataFrame(data)
  df['Images'] = df.Filenames.map(lambda f: get_thumbnail(f))
  df=df[['Filenames','Images','Actual Labels','Predicted Labels']]
  return df

sample_df_halal=sample_results(halal_dir, nama_subfolder_halal, 'Halal')
sample_df_syubhat=sample_results(syubhat_dir, nama_subfolder_syubhat, 'Syubhat')

sample_df=pd.concat([sample_df_halal,sample_df_syubhat])

# displaying PIL.Image objects embedded in dataframe
HTML(sample_df.to_html(formatters={'Images': image_formatter}, escape=False))


# In[22]:


# displaying PIL.Image objects embedded in dataframe
HTML(sample_df.sample(10).to_html(formatters={'Images': image_formatter}, escape=False))


# In[22]:




