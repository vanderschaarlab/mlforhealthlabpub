import sys
from worker import put_blob


url = sys.argv[1]
detailed_result_path = sys.argv[2]

put_blob(url, detailed_result_path)
