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

    signals = ['body_acc_x_train.txt', 
               'body_acc_y_train.txt', 
               'body_acc_z_train.txt', 
               'body_gyro_x_train.txt',
               'body_gyro_y_train.txt',
               'body_gyro_z_train.txt',
               'total_acc_x_train.txt',
               'total_acc_y_train.txt', 
               'total_acc_z_train.txt']
           
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    train_samples = 7352
    test_samples = 2947
    
    for i in range(train_samples):
        x_train.append([])
        
    x_train = load_data_x(train_signals_path, signals, x_train)
    y_train = load_data_y(train_labels_path)
    
    return x_train, y_train
    
if __name__ == "__main__":
    x_train, y_train = create_feature_sets_and_labels()
    print len(x_train), len(y_train)
    print x_train[0]
    print y_train[0]