import os
import argparse

def dff(path):
    items = os.listdir(path)
    for file in items:
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            dff(file_path)
        else:
            if ".exe" in file or ".EXE" in file:
                os.system(f"mv -f \'{file_path}\' ./exe 2>/dev/null || true")

# dff("./unzip")
# print(len(res))

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, help="run on server")
args = parser.parse_args()

if args.f == 'classify':
    items = os.listdir("/home/ubuntu/git_crawler/binary_files")

    for file in items:
        if ".tar" in file:
            os.system(f"cp -f \'./binary_files/{file}\' ./tar")
        elif ".rar" in file:
            os.system(f"cp -f \'./binary_files/{file}\' ./rar")
        elif ".zip" in file:
            os.system(f"cp -f \'./binary_files/{file}\' ./zip")
        elif ".exe" in file:
            os.system(f"cp -f \'./binary_files/{file}\' ./exe")
        else:
            os.system(f"cp -f \'./binary_files/{file}\' ./other")
    
elif args.f == 'unpack':
    # zips = os.listdir("./zip")
    # for zip in zips:
    #     os.system(f"unzip -o ./zip/\'{zip}\' -d ./unpack")
    tars = os.listdir("./tar")
    for tar in tars:
        os.system(f"tar -xvf --overwrite -C ./unpack ./tar/\'{tar}\'")
elif args.f == 'find':
    dff("./unpack")