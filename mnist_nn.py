import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from tqdm import tqdm

input_nodes=784 #because there is 28x28 pixels
hidden_nodes=100
output_nodes=10 #because there is 10 digit
learning_rate=0.3
score_card=[] #for accuracy measurement

# three layers. input, hidden and output layers
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes=input_nodes
        self.hnodes=hidden_nodes
        self.onodes=output_nodes
        self.lr=learning_rate
        
        #creating weight matrix
        self.wih= np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes,self.inodes))
        self.who= np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes,self.hnodes))
        
        #creating sigmoid activation function
        self.activation_function= lambda x: scipy.special.expit(x)
        
        pass
    
    def train(self, input_list, target_list):
        inputs=np.array(input_list, ndmin=2).T   #convert input list to 2D numpy array
        targets=np.array(target_list, ndmin=2).T 
                 
        hidden_inputs= np.dot(self.wih,inputs) #calculate data into hidden layer
        hidden_outputs= self.activation_function(hidden_inputs) #calculate hidden outputs
        final_inputs= np.dot(self.who,hidden_outputs)    #calculate the output layer's input
        final_outputs=self.activation_function(final_inputs) #calculate the final outputs
        
        #finding errors
        output_errors= targets-final_outputs
        hidden_errors= np.dot(self.who.T, output_errors)
        
        #error back propagation, update the weights of the links between layers
        
        self.who += self.lr*np.dot(output_errors*final_outputs*(1.0- final_outputs), np.transpose(hidden_outputs)) 
        self.wih += self.lr*np.dot(hidden_errors*hidden_outputs*(1.0- hidden_outputs), np.transpose(inputs)) 
        pass
    
    def query(self,input_list):
        
        inputs=np.array(input_list, ndmin=2).T  
        hidden_inputs= np.dot(self.wih,inputs)    
        hidden_outputs= self.activation_function(hidden_inputs) 
        final_inputs= np.dot(self.who,hidden_outputs)    
        final_outputs=self.activation_function(final_inputs) 
        
        return final_outputs

n=NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate) #creating an object

# assigning datas to lists
training_data_file=open("mnist_train.csv","r")
training_data_list=training_data_file.readlines()
training_data_file.close()

test_data_file=open("mnist_test_20.csv","r")
test_data_list=test_data_file.readlines()
test_data_file.close()

#training the data list, the accuracy rate is 0.94 for 1 epoch, so 1 epoch is sufficient.
for record in tqdm(training_data_list):
    all_values= record.split(",")
    input=(np.asfarray(all_values[1:])/255*0.99)+0.01 #scaling values and shifting them
    targets=np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])]=0.99
    n.train(input,targets)

#testing the data list    
for record in tqdm(test_data_list):
    all_values=record.split(",")
    correct_label= int(all_values[0])
    test_data=(np.asfarray(all_values[1:])/255*0.99)+0.01
    result=n.query(test_data)
    label=np.argmax(result)
    
    #accuracy measurement
    if label==correct_label:
        score_card.append(1)
    else:
        score_card.append(0)
    #showing the image and prediction in testing data list   
    image_array=np.asfarray(all_values[1:]).reshape((28,28))
    plt.title("Prediction:"+str(label)+" Correct Num: "+str(correct_label))
    plt.imshow(image_array, cmap="Greys")
    plt.show()

#calculating accuracy
score_card_array= np.asarray(score_card)
accuracy=score_card_array.sum()/score_card_array.size
print(accuracy)
