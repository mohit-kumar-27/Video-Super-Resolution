"""
Kaggel Download Algorithm 

Viemo Dataset Download Kaggle URl=https://www.kaggle.com/datasets/wangsally/vimeo-90k-1

Note: -Initilize Kaggle Access Key to dirrectory ~./kaggel

"""

import argparse, os, sys, shutil, urllib.request, logger
import glob
from tqdm import tqdm
import zipfile
import kaggle

kaggle.api.authenticate()

DATASET_URL = 'wangsally/vimeo-90k-{i}'
DATA_FOLDER = "./Vimeo90k/"
DATA_FOLDER_EXT = './Vimeo'
FORCE_DOWNLOAD = False ######################################## Try
parser = argparse.ArgumentParser(description='Dataset Fetcher.')
parser.add_argument('-d', '--debug', default=False, action='store_true', help='Print debug spew.')
args = parser.parse_args()


for i in range(1,11,1):
    kaggle.api.dataset_download_files(f'wangsally/vimeo-90k-{i}', path=DATA_FOLDER, unzip=False,quiet=False,force=FORCE_DOWNLOAD)

list_zip=glob.glob(DATA_FOLDER+"/*.zip")

print(f"Total Zip downloaded {list_zip}")
for  i in list_zip:
    logger.info("Extracting: %s", os.path.join(i))
    try:
        folder=os.path.basename(i).split('.')[0]
        if os.path.exists(os.path.join(DATA_FOLDER_EXT,folder)):
            print(f"{folder} Alreadly Exist ")
        else:
            with zipfile.ZipFile(i, 'r') as zipObj:
            # Extract all the contents of zip file in current directory
                zipObj.extractall(DATA_FOLDER_EXT+f'/{folder}')
    except zipfile.BadZipFile:
        # Re-download the file
        logger.errorOut(f"Redownload file:{i}")

list_file=os.listdir(DATA_FOLDER_EXT)

# Check Validity of below code before executing
DEST_PATH = './Vimeo90k'
os.makedirs(os.path.join(DEST_PATH,'sequences'),exist_ok=True)
test_f = open(DEST_PATH+ '/sep_testlist.txt','w')
train_f = open(DEST_PATH+ '/sep_trainlist.txt','w')

list_dir = os.listdir(DATA_FOLDER_EXT)
for main_dir in list_dir:
    sequencesPath = os.path.join(DATA_FOLDER_EXT,main_dir, "sequences")
    videofolder = os.listdir(sequencesPath)
    videofolder.sort()
    for video in videofolder:
        shutil.move(os.path.join(sequencesPath,video),os.path.join(DEST_PATH,'sequences'))
    with open(os.path.join(DATA_FOLDER_EXT,main_dir,'sep_trainlist.txt'),'r') as infile:
        train_f.write(infile.read())
    # test_f.write('\n')
    with open(os.path.join(DATA_FOLDER_EXT,main_dir,'sep_testlist.txt'),'r') as infile:
        test_f.write(infile.read())
test_f.close()
train_f.close()


