import os, shutil
from os import listdir
from os.path import isfile, join
from prettytable import PrettyTable

agdata = []
gclass_names = { 0 : 'm', 1 : 'f'}
gclass = {'m' : 0, 'f' : 1}
aclass = {'(0,18)' : 0 , '(19,100)' : 1}
inaclass = ['(0, 2)', '(4, 6)', '(8, 12)', '(8, 23)', '(15, 20)', '(25, 32)', '(27, 32)', '(38, 42)', '(38, 43)', '(38, 48)', '(48, 53)', '(60, 100)']
bdir1 = '/mnt/c/Users/saura/Documents/Adience/'
bdir3 = '/mnt/c/Users/saura/Documents/Adience/UTKFace/'
flist = ['fold_0_data.txt', 'fold_1_data.txt', 'fold_2_data.txt', 'fold_3_data.txt', 'fold_4_data.txt']
alldata = []

for txt in flist:
	with open(bdir1 + txt,'r') as f:
		lines = f.readlines()[1:]
		for line in lines:
			data = line.strip().split('\t')
			alldata.append([bdir1 + 'aligned/' + data[0] + '/landmark_aligned_face.' + data[2] + '.' + data[1], data[3], data[4]])
files = [f for f in listdir(bdir3) if isfile(join(bdir3, f))]
for file in files:
	index = file.strip().split('_')
	alldata.append([bdir3 + file, index[0], gclass_names[int(index[1])]])
for data in alldata:
	try:
		if data[1] == '(8, 12)' or data[1] == '(4, 6)' or data[1] == '(8, 23)' or data[1] == '(0, 2)':
			data[1] = '(0,18)'
		elif data[1] not in inaclass and data[1] != 'None' and int(data[1]) < 19:
			data[1] = '(0,18)'
		elif data[1] != 'None':
			data[1] = '(19,100)'
		if data[2] != '' and data[1] != 'None' and data[2] != 'u':
			agdata.append((data[0], aclass[data[1]], gclass[data[2]]))
	except Exception as e:
		print e
um, uf, am, af = (0 for i in range(4))
for i in agdata:
	if i[1] == 0:
		if i[2] == 0:
			um += 1
		else:
			uf += 1
	else:
		if i[2] == 0:
			am += 1
		else:
			af += 1
t = PrettyTable(['Age', 'Male', 'Female'])
t.add_row(['UnderAge', um, uf]) 
t.add_row(['AboveAge', am, af])
print t