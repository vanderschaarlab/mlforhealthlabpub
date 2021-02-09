"""
Run this locally to gather the logs from compute VMs.
"""
import os
import datetime
import shutil
import re


def process_file(filepath):
    last_write = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
    formatted = last_write.strftime("%Y-%m-%d_%H-%M-%S")

    content = ""
    with open(filepath, "r") as f:
        for _ in range(100):
            content += f.readline()
    result_list = re.findall(r'^Running (.+) in', content, flags=re.MULTILINE)
    if len(result_list) > 0:
        username_or_vs = result_list[0]
        if "vs" in username_or_vs:
            # Seeker
            username = username_or_vs.split(" ")[0]
        else:
            # Hider
            username = username_or_vs
    else:
        username = "UNKNOWN"

    new_filepath = os.path.join(
        os.path.realpath(os.path.join(os.path.dirname(filepath), "..")), 
        f"{username}__{os.path.basename(filepath).replace('.log', '')}__LastModifiedDateTime-UTC-{formatted}.log")
    print(f"Copying as: {new_filepath}")
    shutil.copyfile(filepath, new_filepath)


def main():
    n_vms = 15  # NOTE: Set this!
    
    rawpath = os.path.realpath("./raw/")
    if not os.path.exists(rawpath):
        os.makedirs(rawpath)
    
    print("Downloading logs...")
    for i in range(1, n_vms + 1):
        command = f"scp -o StrictHostKeyChecking=accept-new -p ubuntu@codalab-worker-{i}.eastus.cloudapp.azure.com:/home/ubuntu/*.log ./raw/"
        print(f"Executing {command} ...")
        os.system(command)
    
    print("Renaming files...")
    directory = rawpath
    for subdir, _, files in os.walk(directory):
        for f in files:
            filepath = os.path.join(subdir, f)
            if filepath.endswith(".log"):
                process_file(filepath)


if __name__ == "__main__":
    main()
