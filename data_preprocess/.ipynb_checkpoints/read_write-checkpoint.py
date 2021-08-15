import numpy as np
import csv
from tqdm import tqdm
import os
import sys

def check_folder_exist(fpath):
    if os.path.exists(fpath):
        print("dir \"" + fpath + "\" existed")
    else:
        try:
            os.mkdir(fpath)
        except:
            print("error when creating \"" + fpath + "\"") 
        

def write_lines(fname, data):
    print("Save data to \"" + fname + "\"")
    with open(fname, "w") as csvFile:
        if len(data) > 0:
            fw = csv.writer(csvFile, delimiter = ",")
            if type(data[0]) == int or type(data[0]) == float:
                for i in tqdm(range(len(data))):
                    fw.writerow([data[i]])
            else:
                for i in tqdm(range(len(data))):
                    fw.writerow(data[i])
            
def read_lines(fname, row_types = [int, int, float], debug = False):
    print("Load data from \"" + fname + "\"")
    with open(fname, "r") as csvFile:
        fr = csv.reader(csvFile, delimiter=",")
        data = []
        for rowStr in tqdm(fr):
            if len(row_types) == 1:
                row = row_types[0](rowStr[0])
            else:
                row = []
                for t in range(len(row_types)):
                    row.append(row_types[t](rowStr[t]))
            data.append(row)
    return data