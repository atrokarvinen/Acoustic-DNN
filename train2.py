import numpy as np
import tensorflow as tf
import read_data
from sklearn.model_selection import train_test_split


data = list(read_data.read_joint_feat_alignment(alidir="mono_ali", set="train_1h", type="mfcc", cmvn=True, deltas=True))
#Here all numpy arrays for each utterance are simply concatenated. If you are training e.g. a RNN this might not be what you want....
X_data = np.concatenate(tuple(x[1] for x in data))
y_data = np.concatenate(tuple(x[2] for x in data))
res_kinds=127
res_num=len(y_data)
y_data_onehot=np.zeros((res_num,res_kinds))
#Remove original numpy matrices to save memory
del data

one_hot=np.zeros((1,res_kinds))
for i in range(res_num):
    one_hot[0,y_data[i]]=1
    y_data_onehot[i,:]=one_hot
    one_hot[0,y_data[i]]=0

y_data=y_data_onehot
print(X_data.shape)
print(y_data.shape)
X_train, X_vali, y_train, y_vali=train_test_split(X_data,y_data,test_size=0.33,random_state=20)
# Now you can train (and save) your model
trainData=X_train
trainLabel=y_train.reshape(len(y_train),-1)
valiData=X_vali
valiLabel=y_vali.reshape(len(y_vali),-1)

RANDOM_SEED = 10
tf.set_random_seed(RANDOM_SEED)

def initial_weights(shape):
    w=tf.random_normal(shape,stddev=0.1)
    return tf.Variable(w)
def initial_bias(shape):
    b=tf.constant(0.1,shape=[shape])
    return tf.Variable(b)

#suppose two layers, output doesn't use softmax
def forward_propagation(X,w1,w2,w3,b1,b2,b3):
    h1=tf.matmul(X,w1)+b1
    y1=tf.nn.sigmoid(h1)
    h2=tf.matmul(y1,w2)+b2

    #y2=tf.nn.sigmoid(h2)
    #h3=tf.matmul(y2,w3)+b3
    return h2

#this condition outcome is form of [0,0,0,0,0,...1,0...]
def main():
    feature_dimension=trainData.shape[1]
    result_dimension=trainLabel.shape[1]
    hidden_layer_size=256
    hidden_layer_size2=123
    layers = [10,15,20,25,30,35,50,100,200] #ADD
    #l1 = 256
    #l2 = 100
    out = "" #ADD
    #REF ACC: 0.417546
    for hidden_layer_size in layers:
        with tf.Graph().as_default():
            input_data=tf.placeholder("float",shape=[None,feature_dimension])
            output_data=tf.placeholder("float",shape=[None,result_dimension])
            w1=initial_weights((feature_dimension,hidden_layer_size))
            b1=initial_bias(hidden_layer_size)   
            w2=initial_weights((hidden_layer_size,result_dimension))
            b2=initial_bias(result_dimension)

            #THESE ARE FOR ADDITIONAL HIDDEN LAYER
            
            #w1=initial_weights((feature_dimension, l1))
            #b1=initial_bias(l1)
            #w2=initial_weights((l1,l2))
            #b2=initial_bias(l2)
            #w3=initial_weights((l2,result_dimension))
            #b3=initial_bias(result_dimension)
            #forward propagation
            b3 = 1
            w3 = 1
            y2=forward_propagation(input_data,w1,w2,w3,b1,b2,b3)
            prediction=tf.argmax(y2,1)
            #backward propagation
            cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y2,output_data))
            optimizer=tf.train.AdamOptimizer()
            minimize=optimizer.minimize(cost)
            #Run our train optimization in session
            with tf.Session() as sess:
                init_op=tf.initialize_all_variables()
                sess.run(init_op)
                batch_size=1000
                number_of_batch=len(trainData)//batch_size
                number_of_epoch=5
                out += ("\nNumber of hidden layers: %d\n" % hidden_layer_size) #ADD
                for epoch in range(number_of_epoch):
                    #no shuffle currently
                    for i in range(number_of_batch):
                        inData=trainData[i*batch_size:(i+1)*batch_size]
                        outData=trainLabel[i*batch_size:(i+1)*batch_size]
                        sess.run(minimize,feed_dict={input_data:inData, output_data:outData})
                    pre_result=sess.run(prediction,feed_dict={input_data:valiData})
                    validation_accuracy=np.mean(pre_result==np.argmax(valiLabel,1))
                    out += ("You are now at epoch %d!\n" % epoch)  #ADD
                    out += ("The accuracy of validation part is: %f\n" % validation_accuracy) #ADD 
                    print("You are now at epoch %d!" % epoch)
                    print("The accuracy of validation part is: %f" % validation_accuracy)
                    print("Task over. Model has been built.")

    #THIS SAVES TO A FILE "output.txt" THE RESULTS
    with open("output.txt",'w') as result: #ADDED
        result.write(out + '\n')
                #Save=tf.train.Saver()
                #save_path=Save.save(sess,"train_20h_model.ckpt")

if __name__ == '__main__':
    main()   

    
    
    
    
    


