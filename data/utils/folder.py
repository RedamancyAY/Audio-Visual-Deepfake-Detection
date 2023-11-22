import os


# + tags=[]
def check_dir(file):
    dirs = os.path.split(file)[0]
    if not os.path.exists(dirs):
        os.makedirs(dirs)