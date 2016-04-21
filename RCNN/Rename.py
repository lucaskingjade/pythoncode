import os

path = "C:/Users/Jing/Documents/GitHub/data/tennis/3/"
#first collect all files that start with a number and end with .png
my_files = [f for f in os.listdir(path) if f[0].isdigit() and f.endswith(".png")]
#sort them based on the number  
sorted_files = sorted(my_files,key=lambda x:int(x.split(".")[0])) # sort the file names by starting number
#rename them sequentially
i = 0
for fn in sorted_files: #thanks wim
    print(my_files[i])
    os.rename(path+my_files[i],path+str(i)+".png")
    i += 1 