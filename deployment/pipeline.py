import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import json
import pydicom
import csv
import pandas as pd
import io

watch_dir = 'ingress/'
ingress_folder = 'ingress/'
receive_folder = 'receive/'
output_folder = 'output/'
structured_ingress = 'structured_ingress.csv'
patient_status = 'patient_status.csv'


'''
    This function will be called only when new event has been created
    It will generate new path for the dicom file, copy it to structured_ingress folder; update both csv file. 
    input: income dicom file (in ingress)
    output: path to store the file (in structured_ingress)
'''
def structure(filename):
    # get info from dicom
    filename = 'ingress/'+filename
    exp = pydicom.dcmread(filename).SeriesDescription
    patient = pydicom.dcmread(filename).PatientID
    ser_num = pydicom.dcmread(filename).SeriesNumber

    # new file path
    n_name = '/'.join([str(patient).replace(" ", "_"), str(ser_num).replace(" ", "_"), str(exp).replace(" ", "_"), filename.replace("ingress/", "")])
    n_name = "structured_ingress/" + n_name

    # check dir
    dir_name, _ = os.path.split(n_name)
    os.makedirs(dir_name, exist_ok = True)

    # copy the file to new path, old one will still be in in-gress
    os.system(f'cp {filename} {n_name}')
    print('Done Copying')

    # add new item to csv file
    with open(structured_ingress, 'a') as f:
        f.write(','.join([filename, n_name, 'uninf']))
        f.write('\n')
        f.close()
    print('Added new item')

    # update last modified to patient csv
    data = dict(csv.reader((open(patient_status, 'r'))))
    data[str(patient)] = time.time()
    print(data)
    with open(patient_status, 'w') as f:
        for key in data.keys():
            f.write("%s,%s\n"%(key,data[key]))

    return n_name

'''
    This function checks status of each patients in the patient dictionary, and determine which patients are stable
    output: a list of stable patients 
'''
def checkStatus():
    with io.open(patient_status, 'r') as f:
        data = dict(csv.reader(f))

    stable = set()
    for i, (patient, last_modified) in enumerate(data.items()):
        if i == 0:
            continue
        if time.time() - float(last_modified) >= 60 * 10:
            stable.add(patient)
        else:
            continue
    print('List of stable patient:', list(stable))
    return list(stable)

'''
    This function find all the stable patients and move all relevant files from ingress to temp folder. 
    output an update list of files to be inferenced 
'''
def findFiles():
    stable_list = checkStatus()
    # go to folder of ingress
    ingress_list = os.listdir(ingress_folder)
    # find all relevant patient 
    if not os.path.exists('temp/'):
        os.mkdir('temp/')

    update_list = []
    for i in stable_list:
        # process the patient
        # find patients first
        for filename in ingress_list:
            curr_patient = pydicom.dcmread('ingress/'+filename).PatientID
            if str(curr_patient) == i:
                # move from ingress to temp, update structured csv
                os.system(f'cp ingress/{filename} temp/{filename}')
                update_list.append(filename)
                os.remove('ingress/'+filename)
    return update_list

def callInference():

    os.system('python3 /opt/pkd-data/akshay-code/adpkd-segmentation-pytorch/adpkd_segmentation/inference/inference.py --config_path /opt/pkd-data/akshay-code/adpkd-segmentation-pytorch/checkpoints/inference.yml')

    print('call inference here')

def updateStatus(update_list):
    if len(update_list) == 0:
        return
    df = pd.read_csv(structured_ingress)
    df.head()
    for filename in update_list:
        df.loc[df['file']=="ingress/"+filename, "status"] = 'inf'
    df.to_csv(structured_ingress, index=False)


'''
This part will watch once any file is coming in; it will call the structure;
'''
class OnMyWatch:
    # Set the directory on watch
    watchDirectory = watch_dir

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.watchDirectory, recursive = True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Observer Stopped")

        self.observer.join()


class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None
        elif event.event_type == 'created':
            # Event is created, can process it now
            print("Watchdog received created event - % s." % event.src_path)
            filename = event.src_path.split('/')[-1]
            # structure the file first
            structure(filename)
            # find files 
            updatelist = findFiles()
            print('update list:', updatelist)
            # call inference
            callInference()
            updateStatus(updatelist)


if __name__ == '__main__':
    watch = OnMyWatch()
    watch.run()