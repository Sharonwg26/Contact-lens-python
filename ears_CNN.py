# -*- coding: utf-8 -*-
import numpy as np #支援矩陣運算
import os
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from PIL import Image

train_path="./test_img"
test_path="./train_img"

# data_x(image) 與 data_y(label) 前處理
def data_preprocess(datapath):
    img_row,img_col=640,480 #定義圖片大小
    data_x=np.zeros((640,480)).reshape(1,640,480) #儲存圖片
    data_y=[] #紀錄label
    count=0 # 紀錄圖片張數
    num_class=2 # 種類10種
    # 讀取test 資料夾內的檔案
    for root,dirs,files in os.walk(datapath):
        for f in files:
            label=int(root.split('\\')[1]) # 取得label
            data_y.append(label)
            fullpath=os.path.join(root,f) # 取得檔案路徑
            img=Image.open(fullpath) # 開啟image 
            img=img.resize((img_row,img_col))
            img = img.convert("L")
            img = np.array(img)/255
            img = np.reshape(img,(1,img_row,img_col)) # 作正規化與reshape
            data_x=np.vstack((data_x,img))
            count+=1
    data_x=np.delete(data_x,[0],0) # 刪除np.zeros
    # 調整資料格式
    data_x=data_x.reshape(count,img_row,img_col,1)
    data_y=np_utils.to_categorical(data_y,num_class) # 將label轉成one-hot.encoding
    return data_x,data_y

# 顯示訓練圖片
def show_train(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.xlabel('Epoch')
    plt.ylabel(train)
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

data_x, data_y=data_preprocess(train_path)

model=Sequential() # 建立模型
model.add(Conv2D(filters=32,kernel_size=(2,2),padding='same',input_shape=(640,480,1),activation='relu')) # 建立卷基層
model.add(MaxPooling2D(pool_size=(2,2))) # 建立池化層
# model.add(Dense(units=512,input_dim=784,activation='relu')) # 建立全連接層
model.add(Conv2D(filters=64,kernel_size=(2,2),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dense(units=256,activation='relu')) # 建立全連接層
model.add(Dropout(0.25)) # Dropout隨機斷開輸入神經元，防止過度擬合，比例0.25
model.add(Conv2D(filters=128,kernel_size=(2,2),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dense(units=128,activation='relu')) # 建立全連接層
model.add(Dropout(0.1)) # Dropout隨機斷開輸入神經元，防止過度擬合，比例0.25
model.add(Flatten()) # 多維輸入一維化
model.add(Dropout(0.25))
model.add(Dense(units=2,activation='softmax')) # 使用 softmax,將結果分類 units=10,10類


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy']) #損失函數、優化方法、成效衡量
train_history=model.fit(data_x, data_y,batch_size=32,epochs=200,verbose=1, validation_split=0.1)
show_train(train_history,'accuracy','val_accuracy')
show_train(train_history,'loss','val_loss')

test_x,test_y=data_preprocess(test_path)

score = model.evaluate(test_x,test_y, verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])