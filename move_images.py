
import os

folders = ['AlsoBuses', 'BeautifulBusesYo'] #Folders where files are saved
path = 'C:\Users\user\Desktop\OpenLabelling\images' #images path for Openlabelling github folder(images is for pics,class_list.txt are list of classes)


n = 0 #This is just a counter for parsing through images in each folder
for folder in folders:
    for image in os.scandir(folder): #scanning directory named folder,folder is list of folder names where images are stored
        n+=1
        os.rename(image.path, os.path.join(path, '{:06}.jpg'.format(n)))