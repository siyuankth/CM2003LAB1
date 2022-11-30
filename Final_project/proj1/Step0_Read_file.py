import glob
import numpy as np
import re
import os

############################Get the array
# def get_numbers_from_filename(filename):
#     temp = int(re.search(r'\d+', filename).group(0))
#     return temp
#
# def LoadTxtMethod(filename):
#     Data = []
#     i = 0
#     suture_nr = get_numbers_from_filename(filename)
#     for line in open(filename):
#         result = []
#         line = line.strip()
#         if not len(line) or line.startswith('#'):
#             continue
#         data = line.split(',')
#         if (data[0] == ''):
#             continue
#         i = i + 1
#         node_nr = i
#         data_x, data_y, data_z = float(data[1]), float(data[2]), float(data[3])
#         result.append(suture_nr)
#         result.append(node_nr)
#         result.append(data_x)
#         result.append(data_y)
#         result.append(data_z)
#         Data.append(result)
#     Data = np.array(Data)
#     row_nr = Data.shape[0]
#     if suture_nr == 1 or suture_nr == 2 or suture_nr == 7:
#         normal_row_nr = 100
#     if suture_nr == 3 or suture_nr == 4:
#         normal_row_nr = 200
#     if suture_nr == 5 or suture_nr == 6:
#         normal_row_nr = 50
#     if row_nr !=normal_row_nr:
#         print('There is an error for reading file Suture_%d,the shape of number is %d' %(suture_nr,row_nr))
#         print('The filename is %s' %(filename))
#         exit(0)
#     return Data
# myfiledirectory = 'X:/Siyuanch/Project2/Reference/Landmark/'
# os.chdir(myfiledirectory)
# ALL = []
# j = 1
# print(os.listdir(myfiledirectory))
# for filename in os.listdir(myfiledirectory):
#     if j == 1:
#         temp = filename
#         data = LoadTxtMethod(temp)
#         ALL = data
#         j = j + 1
#         print(j)
#     else:
#         temp = filename
#         data = LoadTxtMethod(temp)
#         ALL = np.vstack((ALL,data))
#         j = j + 1
#         print(j)
#
# AA = np.array(ALL)

###################Get the long list
def get_numbers_from_filename(filename):
    temp = int(re.search(r'\d+', filename).group(0))
    return temp

def LoadTxtMethod(filename,result,j):
    Data = []
    i = 0
    suture_nr = get_numbers_from_filename(filename)
    for line in open(filename):
        # result = []
        line = line.strip()
        if not len(line) or line.startswith('#'):
            continue
        data = line.split(',')
        if (data[0] == ''):
            continue
        i = i + 1
        data_x, data_y, data_z = float(data[1]), float(data[2]), float(data[3])
        result.append(data_x)
        result.append(data_y)
        result.append(data_z)
        Data.append(result)
    Data = np.array(Data)
    row_nr = Data.shape[0]
    if suture_nr == 1 or suture_nr == 2 or suture_nr == 7:
        normal_row_nr = 100
    if suture_nr == 3 or suture_nr == 4:
        normal_row_nr = 200
    if suture_nr == 5 or suture_nr == 6:
        normal_row_nr = 50
    if row_nr !=normal_row_nr:
        print('There is an error for reading file Suture_%d,the shape of number is %d' %(suture_nr,row_nr))
        print('The filename is %s' %(filename))
        print('The filefolder is %s' %(j))
        exit(0)

    return result
# myfiledirectory = 'X:/Siyuanch/Project2/Reference/TRY/'
# Working folder
myfiledirectory = 'X:/Siyuanch/Project2/Step3_GetLandmarks/'
os.chdir(myfiledirectory)
j = 1  #suture_nr counter
jj = 1 #file_nr counter
ALL = []
for filename in os.listdir(myfiledirectory):
    Record_nr = get_numbers_from_filename(filename)
    Data_path = myfiledirectory + '%s'%(Record_nr) + '/'
    os.chdir(Data_path)
    path = os.getcwd()
    fcsv_files = glob.glob(os.path.join(path,"*.fcsv"))
    for f in fcsv_files:
        suturename = f.split("\\")[-1]
        if j == 1:
            temp = suturename
            if j != get_numbers_from_filename(temp):
                print('Error: The order of the files name is wrong in %s'%(Data_path))
                exit(0)
            data = LoadTxtMethod(temp,[Record_nr],jj)
            j = j + 1
        else:
            temp = suturename
            if j != get_numbers_from_filename(temp):
                print('Error: The order of the files name is wrong in %s'%(Data_path))
                exit(0)
            data = LoadTxtMethod(temp,data,jj)
            j = j + 1
    j = 1 # Re-initialize suture_nr counter
    ALL.append(data)
    jj = jj + 1
print('The data has been loaded successfully, in total of %d group of data'%(jj))

import csv

# field names

fields = ['Record_ID']
len = int((len(data)-1)/3)
for i in range(len):
    ii = i + 1
    fields.append('p%d_x' %(ii))
    fields.append('p%d_y' %(ii))
    fields.append('p%d_z' %(ii))


# data rows of csv file
with open('X:/Siyuanch/Project2/DATA.csv', 'w',newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(ALL)


FILE = np.array(data)

