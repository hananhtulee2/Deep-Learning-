import numpy as np 
import matplotlib.pyplot as plt  
import tensorflow as tf
# khai bao cac thu vien su dung trong bai 

print("Load MNIST Database")
# Tai dataset MNIST
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)= mnist.load_data()

# kich thuoc cua  buc anh 28x28 pixel bang cach su dung np.shape , du lieu se bien thanh mot mang 2 chieu voi kich 
# thuoc la (so luong mau, 784) tuc la moi buc anh duoc bien thanh vecto kich thuoc 784 

# thong so /255 la viec chuan hoa du lieu de moi gia trị pixel nam trong khoang 0->1 , giup mo hinh hoc tot hon 
x_train=np.reshape(x_train,(60000,784))/255.0
x_test= np.reshape(x_test,(10000,784))/255.0


# chuyen doi cac nhan tu dang dang so nguyen thanh dang one hot encoding 
# Vi du du lieu co nhan la so 3 thi sau khi dung one hot encoding thi se thanh 000100000, 
# Vi du tren nhan la so 3 nen o vi tri so 4 (tuc la 0 1 2 3 ..) trong day se len muc 1 con lai xuong muc 0 tuong tu cac truong hop con lai
y_train = np.matrix(np.eye(10)[y_train])
y_test = np.matrix(np.eye(10)[y_test])
print("----------------------------------")
print(x_train.shape)   # In ra kich thuoc du lieu train 
print(y_train.shape)    # In ra nhan cua du lieu train o tren 

# Khai bao cac ham su dung trong bai : sigmoid , relu , softmax
def sigmoid(x):
    return 1./(1.+ np.exp(-x))
def softmax(x):
    return np.divide(np.matrix(np.exp(x)),np.mat(np.sum(np.exp(x),axis=1)))
def relu(x):
    return np.maximum(x, 0)

# Dao ham cua ham relu 
def relu_derivative(x):
    return np.where(x <= 0, 0, 1)

# lan truyen tien 
def Forwardpass(X, Wh1, bh1, Wh2, bh2, Wo, bo):
    # Tinh toan dau ra cho ngo ra hidden layer 1 dung ham relu
    zh1 = X @ Wh1.T + bh1
    a1 = relu(zh1)
    
    # Tinh toan dau ra cho ngo ra hidden layer 2 dung ham sigmoid
    zh2 = a1 @ Wh2.T + bh2
    a2 = sigmoid(zh2)
    
    # Tinh toan dau ra cho lop ngo ra dung ham softmax 
    z = a2 @ Wo.T + bo
    o = softmax(z)
    
    return o

# tinh toan do chinh xac cho model sau khi build 
def AccTest(label,prediction):    
    OutMaxArg=np.argmax(prediction,axis=1) # tim vi tri gia tri lon nhat theo hang trong ma tran predict => nhan cua du doan 
    LabelMaxArg=np.argmax(label,axis=1) # tim vi tri gia tri lon nhat theo hang trong ma tran label => nhan thuc te cua du lieu 
    Accuracy=np.mean(OutMaxArg==LabelMaxArg) # tinh ti le trung binh giua nhan du doan va nhan thuc te 
    return Accuracy    # gia tri tra ve la do chinh xac 

# setup mot so thong so cho model nhu learningrate , epoch,..
learningRate = 0.5  # toc do hoc 
Epoch=50 # so lan lap qua tap du lieu huan luyen 
NumTrainSamples=60000    
NumTestSamples=10000

NumInputs=784  # vi da duỗi cac buc anh 28x28 thanh vecto 784 nen dau vao input la 784 
NumHiddenUnits=512  
NumClasses=10   # so luong ngo ra la 10 tuong ung voi nhan dang 10 chu so 0 den 9 


# Hidden layer thu nhat relu
Wh1 = np.matrix(np.random.uniform(-0.5, 0.5, (NumHiddenUnits, NumInputs)))
bh1 = np.random.uniform(0, 0.5, (1, NumHiddenUnits)) 
dWh1 = np.zeros((NumHiddenUnits, NumInputs))
dbh1 = np.zeros((1, NumHiddenUnits))

# Hidden layer thu hai sigmoid 
Wh2 = np.matrix(np.random.uniform(-0.5, 0.5, (NumHiddenUnits, NumHiddenUnits)))
bh2 = np.random.uniform(0, 0.5, (1, NumHiddenUnits))
dWh2 = np.zeros((NumHiddenUnits, NumHiddenUnits))
dbh2 = np.zeros((1, NumHiddenUnits))

# Output layer softmax 
Wo = np.random.uniform(-0.5, 0.5, (NumClasses, NumHiddenUnits))
bo = np.random.uniform(0, 0.5, (1, NumClasses))
dWo = np.zeros((NumClasses, NumHiddenUnits))
dbo = np.zeros((1, NumClasses))



from IPython.display import clear_output
loss = []   # luu gia tri cho ham mat mac de cap nha trong so   
Acc = []    # luu do chinh xac model 
Batch_size = 200  # moi lan chi lay 200 mau tu 60000 mau trong tep du  train 
Stochastic_samples = np.arange(NumTrainSamples) # tao ra 1 mang chua cac chi so cua cac mau trong tep train
for ep in range(Epoch):
    np.random.shuffle(Stochastic_samples)  # xao tron thu tu cua cac mau trong tep train , tang muc do chan thuc moi lan hoc 
    for ite in range(0, NumTrainSamples, Batch_size):  
    
        Batch_samples = Stochastic_samples[ite:ite+Batch_size] # mang luu cac chi so cua mau du lieu trong batch hien tai 
        x = x_train[Batch_samples, :]  # chon ra cac chi so cua batch hien tai tu tep du lieu x_train 
        y = y_train[Batch_samples, :] # chon ra cac nhan cua batch hien tai tu tep y_train  
        
        # 1st layer  
        zh1 = x @ Wh1.T + bh1
        a1 = relu(zh1)
        
        # 2nd layer 
        zh2 = a1 @ Wh2.T + bh2
        a2 = sigmoid(zh2)
        
        # output layer 
        z = a2 @ Wo.T + bo
        o = softmax(z)
        
        # tinh toan ham mat mat 
        loss.append(-np.sum(np.multiply(y, np.log10(o))))
        
        
        # dao ham cua dL/dWz 
        d = o - y
        
        # chain rule o layer 2 
        dh2 = d @ Wo   
        dhs2 = np.multiply(np.multiply(dh2, a2), (1 - a2))  # giong voi chung minh trong giay A4 da ghi
        #  chain rule o layer 1 
        dh1 = dhs2 @ Wh2
        dhs1 = np.multiply(dh1, relu_derivative(a1))  # giong voi chung minh trong giay A4 da ghi 
        
        # output layer
        dWo = np.matmul(np.transpose(d), a2)
        dbo = np.mean(d)
        
        #  layer 2
        dWh2 = np.matmul(np.transpose(dhs2), a1)
        dbh2 = np.mean(dhs2)
        
        # layer 1
        dWh1 = np.matmul(np.transpose(dhs1), x)
        dbh1 = np.mean(dhs1)
        
        # cap nhat trong so trong mang 
        Wo = Wo - learningRate * dWo / Batch_size
        bo = bo - learningRate * dbo
        Wh2 = Wh2 - learningRate * dWh2 / Batch_size
        bh2 = bh2 - learningRate * dbh2
        Wh1 = Wh1 - learningRate * dWh1 / Batch_size
        bh1 = bh1 - learningRate * dbh1
        
        # goi cac ham da giai thich o tren de tinh toan do chinh xac cho tung vong lap
        prediction = Forwardpass(x_test, Wh1, bh1, Wh2, bh2, Wo, bo)
        Acc.append(AccTest(y_test, prediction))
        clear_output(wait=True)
        plt.plot([i for i, _ in enumerate(Acc)], Acc, 'o')  # ve tung diem acc theo moi vong lap 
        plt.show() # hien thi hinh anh 
        
    print('Epoch:', ep)  
    print('Accuracy:', AccTest(y_test, prediction))


