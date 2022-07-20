'''
Modified from SCN code
This script generate a txt file with a asymetrical noisy labels for RafDB
# 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
'''
import random
noise_ratio=0.6
new_file = open("inject{}noise_asym_cmat.txt".format(noise_ratio),"w")
noise_dict = { 1 : 6, 2: 1, 3:6, 4:7,   5:7, 6:4,7:5}
with open("/mnt/Data/tohar/raf-basic/basic/EmoLabel/list_patition_label.txt","r") as file:
	for line in file:
		line = line.strip()
		img_path, label = line.split(' ', 1)
		number = random.uniform(0,1)
		#new_label = random.randint(1,7) #labels are between 1 to 7 range
		if number <= noise_ratio:
			#import pdb; pdb.set_trace()
			new_label = noise_dict[int(label) ]
			if int(label) == 7:
				new_label = 1

			new_file.write(img_path + ' ' + str(new_label) +'\n')

		else:
			new_file.write(img_path + ' ' + str(label) +'\n')
