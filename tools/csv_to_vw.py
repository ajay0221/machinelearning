import csv
from ConfigParser import ConfigParser

if __name__ == "__main__":
    #config_filename = sys.argv[1]
    config_filename = "config/params.ini"
    config = ConfigParser()
    config.read(config_filename)

    config_dict = dict()
    for (key,value) in config.items('varlist'):
        config_dict[key] = [i.strip() for i in value.strip().split(',')]

    input_filename = config_dict['input_file']
    with open(input_filename, 'rb') as f:
        reader = csv.reader(f)
        headers = reader.next()
        print headers
        print
        for row in reader:
            print row