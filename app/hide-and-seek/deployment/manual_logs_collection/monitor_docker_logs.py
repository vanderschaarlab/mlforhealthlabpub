"""
Run this on VMs as:
```bash
python3 -u monitor_docker_logs.py &> monitor.txt & 
disown
```
"""
import os
import subprocess
import time


LOGS_DIR = "/home/ubuntu/"  # TODO: Set as needed.
SIGNATURE = "python3 /tmp/codalab"  # TODO: Set as needed.
SLEEP_TIME = 60  # TODO: Set as needed.


def main():
    
    while True:

        titles = ["CONTAINER ID", "IMAGE", "COMMAND", "CREATED", "STATUS", "PORTS", "NAMES"]
        command_col = 2
        name_col = 6
        
        result = subprocess.run(['docker', 'ps', '--no-trunc'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        lines = [x for x in result.split("\n") if x != ""]
        line_0 = lines[0]
        sep_indices = [line_0.find(x) for x in titles]
        data = list()
        for idx, line in enumerate(lines):
            data.append(list())
            for jdx, s in enumerate(sep_indices[:-1]):
                data[idx].append(line[s:sep_indices[jdx + 1]].strip())
            data[idx].append(line[sep_indices[-1]:].strip())
        matched_names = list()
        for line in data:
            if SIGNATURE in line[command_col]:
                matched_names.append(line[name_col])
        
        existing_logs_filenames = list()
        for subdir, _, files in os.walk(os.path.realpath(LOGS_DIR)):
            for f in files:
                filepath = os.path.join(subdir, f)
                if filepath.endswith(".log"):
                    existing_logs_filenames.append(f)
        # print(f"Existing logs: {existing_logs_filenames}")
        
        for name in matched_names:
            logfile_expected = name + ".log"
            if logfile_expected not in existing_logs_filenames:
                print(f"Spawning log monitoring for: {name} (to {logfile_expected})")
                with open(logfile_expected, "wb") as f:
                    subprocess.Popen(["docker", "logs", "-f", name], stdout=f, stderr=f)
        
        # print(f"Waiting {SLEEP_TIME}s...")
        time.sleep(SLEEP_TIME)


if __name__ == "__main__":
    main()
