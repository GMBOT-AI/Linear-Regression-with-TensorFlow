import tensorflow as tf
#initializing w,b
w=tf.Variable(0.0)
b=tf.Variable(0.0)
#training data
train_x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
train_y = tf.constant([2.0, 3.0, 4.0, 5.0, 6.0])
#iterations
training_steps=100
#learning rate
learning_rate=0.01

for i in range(training_steps):
#start recording the steps
    with tf.GradientTape() as tape:
     y_pred = train_x * w + b #making prediction
     error = y_pred-train_y #calculating diff b/w real and predicted values
     loss=tf.reduce_mean(tf.square(error)) #reducing mse

    #evauluating values for gradient
    dW, db=tape.gradient(loss,[w,b])
    #updating values 
    w.assign_sub(dW*learning_rate)
    b.assign_sub(db*learning_rate)

    if i%20==0:
     print("Step {:.3f}, Loss: {:.3f}".format(i, loss))

   
