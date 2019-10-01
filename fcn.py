import numpy as np


class LinearLayer(object):
    """
    Basic Linear layer class. Contains:
    Forward pass
    Backward pass
    Gradient descent 
    setlr: Change learning rate for SGD.
    Ability to cache data for backprop
    Weights initialized by Xavier Initialization
    """
    np.random.seed(45)
    
    def __init__(self,input_size,output_size,learning_rate=0.01):
        self.shape = [input_size,output_size]
        self.W = np.random.normal(0,np.sqrt(1/input_size),(input_size,output_size))*0.1
        self.b = np.zeros([1,output_size])
        self.layer_cache = 0
        self.output = 0
        self.dW = 0
        self.db = 0
        self.dIp =0
        self.lr = learning_rate
   
    def forward(self,input_mat):
        self.output = np.dot(input_mat,self.W)+ self.b
        self.layer_cache = input_mat
        return self.output
    
    def backward(self,dy):
        n = self.layer_cache.shape[1]
        self.dW = np.dot(np.transpose(self.layer_cache),dy)
        self.db = np.sum(dy,0)
        self.dIp = np.dot(dy,self.W.T)
        return self.dIp
   
    def step(self):
        self.W =  self.W - self.lr*self.dW
        self.b =  self.b - self.lr*self.db.reshape(1,-1)
        '''print(self.dW)
        print("differnt----")
        print(self.W)
        print("weight-----")'''
    
    def setlr(self,lr):
        self.lr = lr
            
class ReLu(object):
    """
    ReLu activation contains:
    Forward pass
    Backward Pass
    Ability to set leak if LeakyRelu desired
    Caching ability
    """
    
    def __init__(self):
        self.output = 0
        self.acti_layer_cache = 0
        self.dy = 0
        self.leak = 0
    
    def forward(self,input_mat):
        index = input_mat <0
        self.output = input_mat
        self.output[index] = input_mat[index]*self.leak
        self.acti_layer_cache = input_mat
        return self.output
    
    def backward(self,dz):
        index1 = self.acti_layer_cache >0
        index2 = self.acti_layer_cache <0
        self.dy = self.acti_layer_cache
        self.dy[index1]=1
        self.dy[index2]=self.leak
        self.dy = np.multiply(self.dy,dz)
        return self.dy

class softmax(object):
    """
    Softmax activation containing:
    Forward pass
    Backward pass
    """

   
    def __init__(self):
        self.output = 0
    
    def forward(self,input_mat):
        self.output = np.exp(input_mat)/np.sum(np.exp(input_mat),1).reshape(-1,1)
        return self.output
    
    def backward(self,dy):#Backward pass already computed in cross entropy loss
        return dy  
    
    
class CEloss(object):
    """
    Cross entropy loss containing:
    Forward pass
    Backward pass
    """
    
    def __init__(self):
        self.output = 0
        self.cache = 0
        
    def forward(self,labels,pred):
        n = labels.shape[1]
        self.output = (-1 / n) * np.sum(np.sum(np.multiply(labels,np.log(pred))))
        self.cache = [labels,pred]
        return self.output
   
    def backward(self): #Gives gradient wrt softmax input 
        n = self.cache[0].shape[1]
        dy = 1.0*(self.cache[1]-self.cache[0])
        return dy
    
class Model(object):

    """
    This is the main model class. It forms the core of the architecture
    """
    #Initialize the architecture with dimensions
    def __init__(self,layer_dimensions,learning_rate=1e-3,filesave= "../weights/",load=False):
        self.layers = []
        self.dimensions = layer_dimensions
        for l in range(len(layer_dimensions)-1):
            self.layers.append(LinearLayer(layer_dimensions[l],layer_dimensions[l+1],learning_rate))
            print("Linear added {}".format(self.layers[-1].shape))
            if l  != len(layer_dimensions)-2 :
                self.layers.append(ReLu())
                print("Relu added")
        self.layers.append(softmax())
        print("Softmax added")
        self.Loss = CEloss()
        self.epoch = 0
        self.filesave = filesave
        self.lr = learning_rate
        if load :
            count = 0
            for i in self.layers:
                self.epoch=np.load("../weights/epoch.npy")
                if isinstance(i,LinearLayer):
                    i.W = np.load(self.filesave+"W"+str(count)+".npy")
                    i.b = np.load(self.filesave+"b"+str(count)+".npy")
                    count+=1
                
            
    #Load train and test data
    def LoadData(self,x_train,y_train,x_test,y_test):
        self.X_train = x_train
        self.y_train = y_train
        self.X_test = x_test
        self.y_test = y_test
    
    #Training model. Set epochs and batchsize
    def train(self,epoch=200,batchsize=100):

        epoch_vec = []
        train_acc_vec =[]
        test_acc_vec = []
        for i in range(self.epoch,self.epoch+epoch):
            numbatch = self.X_train.shape[0]//batchsize
            for ibatch in range(numbatch):
                pred = self.forwardPass(self.X_train[ibatch*batchsize:(ibatch+1)*batchsize,:])
                disc_loss = self.computeLoss(self.y_train[ibatch*batchsize:(ibatch+1)*batchsize,:],pred)
                self.backwardPass()
                self.gradientDescent()
                self.reset_fcn()

            self.epoch+=1
            if self.epoch%30== 0:
                self.lr = self.lr/10
                self.change_lr(self.lr)
            print("epoch: {}, Discriminator Loss: {}".format(self.epoch,disc_loss))
            print("Train accuracy is: {}% and Validation accuracy is {}%".format(self.trainAcc(),self.testAcc()))
            epoch_vec.append(self.epoch)
            train_acc_vec.append(self.trainAcc())
            test_acc_vec.append(self.testAcc())
            count = 0
        
        return epoch_vec,train_acc_vec,test_acc_vec
                   
    #Forward pass layer by layer
    def forwardPass(self,input_mat):
        pred = input_mat
        for layer in self.layers:
            pred = layer.forward(pred)
        return pred
    
    #Compute loss 
    def computeLoss(self,labels,pred):
        return self.Loss.forward(labels,pred)
    #Backward pass layer by layer
    def backwardPass(self):
        dy = self.Loss.backward()
        """print("Softmax spits out:")
        print(dy)"""
        for i in range(len(self.layers)-2,-1,-1):
            dy = self.layers[i].backward(dy)
            """print(self.layers[i].__class__.__name__+" spits out:")
            print(dy)
        print("------")"""
            
    #Update weights through SGD
    def gradientDescent(self):
        for i in self.layers:
            if isinstance(i,LinearLayer):
                i.step()
    
    def change_lr(self,lr):
        for i in self.layers:
            if isinstance(i,LinearLayer):
                i.setlr(lr)
        
    #Calculate train accuracy
    def trainAcc(self):
        pred = self.forwardPass(self.X_train)
        pred_one_hot = (pred>=np.max(pred,1).reshape(-1,1)+0).astype(float)
        correct = np.multiply(self.y_train==pred_one_hot,self.y_train).sum()
        total = self.y_train.shape[0]
        return 1.*correct/total*100
    
    #Calculate test accuracy
    def testAcc(self):
        pred = self.forwardPass(self.X_test)
        pred_one_hot = (pred>=np.max(pred,1).reshape(-1,1)).astype(float)
        correct = np.multiply(self.y_test==pred_one_hot,self.y_test).sum()
        total = self.y_test.shape[0]
        return 1.*correct/total*100

    #Reset model weights, gradients and cache.
    def reset_fcn(self):
        for i in self.layers:
            if isinstance(i,LinearLayer):
                i.layer_cache = 0
                i.output = 0
                i.dW = 0
                i.db = 0
                i.dIp =0

            if isinstance(i,ReLu):
                i.output = 0
                i.acti_layer_cache = 0
                i.dy = 0


