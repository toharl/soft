'''
Modified from SCN code
This script generates a txt file with a uniform noisy labels for RafDB
# 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
'''
import random
noise_ratio=0.4
new_file = open("inject{}noise.txt".format(noise_ratio),"w")
with open("/mnt/Data/tohar/raf-basic/basic/EmoLabel/list_patition_label.txt","r") as file:
	for line in file:
		line = line.strip()
		img_path, label = line.split(' ', 1)
		number = random.uniform(0,1)
		new_label = random.randint(1,7) #labels are between 1 to 7 range
		if number <= noise_ratio:
			while(1):
				new_label = random.randint(1,7)
				if new_label != int(label):
					new_file.write(img_path + ' ' + str(new_label) +'\n')
					break
		else:
			new_file.write(img_path + ' ' + str(label) +'\n')
