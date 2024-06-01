import numpy as np 
import matplotlib.pyplot as plt  
import tensorflow as tf
from tensorflow import keras

print("Load MNIST Database")
# Model / du lieu bien 
num_classes = 10

# Tai dataset 
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)= mnist.load_data()
# Scale images to the [0, 1] range
x_train=np.reshape(x_train,(60000,784))/255.0
x_test= np.reshape(x_test,(10000,784))/255.0
# Convert class vectors to binary class matrices
y_train = np.matrix(np.eye(10)[y_train])
y_test = np.matrix(np.eye(10)[y_test])
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
print("----------------------------------")
print(x_train.shape)
print(y_train.shape)
def sigmoid(x):
    return 1./(1.+ np.exp(-x))
def softmax(x):
    return np.divide(np.matrix(np.exp(x)),np.mat(np.sum(np.exp(x),axis=1)))
def relu(x):
    return np.maximum(x, 0)

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


def AccTest(label,prediction):    
    OutMaxArg=np.argmax(prediction,axis=1)
    LabelMaxArg=np.argmax(label,axis=1)
    Accuracy=np.mean(OutMaxArg==LabelMaxArg)
    return Accuracy

learningRate = 0.5
Epoch=50
NumTrainSamples=60000
NumTestSamples=10000

NumInputs=784
NumHiddenUnits=512
NumClasses=10
#inital weights
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
loss = []
Acc = []
Batch_size = 200
Stochastic_samples = np.arange(NumTrainSamples)
for ep in range(Epoch):
    np.random.shuffle(Stochastic_samples)
    for ite in range(0, NumTrainSamples, Batch_size):  
    
        Batch_samples = Stochastic_samples[ite:ite+Batch_size]
        x = x_train[Batch_samples, :]
        y = y_train[Batch_samples, :]
        

        zh1 = x @ Wh1.T + bh1
        a1 = relu(zh1)
        
   
        zh2 = a1 @ Wh2.T + bh2
        a2 = sigmoid(zh2)
        
    
        z = a2 @ Wo.T + bo
        o = softmax(z)
        
        # Calculate loss
        loss.append(-np.sum(np.multiply(y, np.log10(o))))
        
    
        d = o - y
        
        # đạo hàm của sigmoid
        dh2 = d @ Wo
        dhs2 = np.multiply(np.multiply(dh2, a2), (1 - a2))
        # đạo hàm của relu
        dh1 = dhs2 @ Wh2
        dhs1 = np.multiply(dh1, a1 > 0)
        
        # U output layer
        dWo = np.matmul(np.transpose(d), a2)
        dbo = np.mean(d)
        
        #  layer 2
        dWh2 = np.matmul(np.transpose(dhs2), a1)
        dbh2 = np.mean(dhs2)
        
        # layer 1
        dWh1 = np.matmul(np.transpose(dhs1), x)
        dbh1 = np.mean(dhs1)
        
        # cap nhat 
        Wo = Wo - learningRate * dWo / Batch_size
        bo = bo - learningRate * dbo
        Wh2 = Wh2 - learningRate * dWh2 / Batch_size
        bh2 = bh2 - learningRate * dbh2
        Wh1 = Wh1 - learningRate * dWh1 / Batch_size
        bh1 = bh1 - learningRate * dbh1
        
    
        prediction = Forwardpass(x_test, Wh1, bh1, Wh2, bh2, Wo, bo)
        Acc.append(AccTest(y_test, prediction))
        clear_output(wait=True)
        plt.plot([i for i, _ in enumerate(Acc)], Acc, 'o')
        plt.show()
        
    print('Epoch:', ep)
    print('Accuracy:', AccTest(y_test, prediction))


