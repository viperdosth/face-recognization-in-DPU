import os 

a = open("calib.txt","w")
for path,subdirs,files in os.walk('./calib'):
    for filename in files:
        
        a.write(str(filename)+'\n')