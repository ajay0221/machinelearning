import tensorflow as tf
import numpy as np

def load_data_x(train_signals_path, signals, data):
    for signal in signals:
        filename = train_signals_path + signal
        with open(filename,'r') as f:
            lines = f.readlines()
            k = 0
            for line in lines:
                data[k] = data[k] + line.strip().replace('  ',' ').split(' ')
                k += 1
    return data

def load_data_y(path):
    data = []
    with open(path,'r') as f:
        lines = f.readlines()
        for line in lines:
            j = int(line.strip()) - 1
            temp = [0, 0, 0, 0, 0, 0]
            temp[j] = 1
            data.append(temp)
    return data

def create_feature_sets_and_labels():
    folder_path = "/Users/ajay/wit/har/data/"

    train_signals_path = folder_path + "train/inertial_signals/"
    train_labels_path = folder_path + "train/y_train.txt"

    test_signals_path = folder_path + "test/inertial_signals/"
    test_labels_path = folder_path + "test/y_test.txt"

    train_signals = ['body_acc_x_train.txt', 
                     'body_acc_y_train.txt', 
                     'body_acc_z_train.txt', 
                     'body_gyro_x_train.txt',
                     'body_gyro_y_train.txt',
                     'body_gyro_z_train.txt',
                     'total_acc_x_train.txt',
                     'total_acc_y_train.txt', 
                     'total_acc_z_train.txt']
                     
    test_signals = ['body_acc_x_test.txt', 
                    'body_acc_y_test.txt', 
                    'body_acc_z_test.txt', 
                    'body_gyro_x_test.txt',
                    'body_gyro_y_test.txt',
                    'body_gyro_z_test.txt',
                    'total_acc_x_test.txt',
                    'total_acc_y_test.txt', 
                    'total_acc_z_test.txt']
           
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    train_samples = 7352
    test_samples = 2947
    
    for i in range(train_samples):
        x_train.append([])
        
    x_train = load_data_x(train_signals_path, train_signals, x_train)
    y_train = load_data_y(train_labels_path)
    
    for i in range(test_samples):
        x_test.append([])
    
    x_test = load_data_x(test_signals_path, test_signals, x_test)
    y_test = load_data_y(test_labels_path)
    return x_train, y_train, x_test, y_test
    
x_train, y_train, x_test, y_test = create_feature_sets_and_labels()
print 'Number of traingin samples : ', len(x_train)
print 'Number of testing samples : ', len(x_test)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_nodes_hl4 = 500

n_classes = 6
batch_size = 100

x = tf.placeholder('float', [None, len(x_train[0])])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(x_train[0]), n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    hidden_4_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}
                      
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.relu(l4)
    
    output = tf.add(tf.matmul(l4, output_layer['weights']), output_layer['biases'])
    return output
    
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    hm_epochs = 300
    max_accuracy = 0.0
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            
            i = 0
            while i < len(x_train):
                start = i
                end = i + batch_size

                batch_x = np.array(x_train[start:end])
                batch_y = np.array(y_train[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            print 'Epoch ', (epoch+1), 'completed out of', hm_epochs, 'loss :', epoch_loss
        
        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print 'Accuracy :', accuracy.eval({x: x_test, y:y_test})
        
train_neural_network(x)