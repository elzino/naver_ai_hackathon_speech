import os
import time

submit_format = "nsml submit team39/sr-hack-2019-50000/"
ARGS = ["31 best0_04362819374052981", "90 model", "73 model", "90 model", "73 model", "90 model"]

time.sleep(60*9)

for i in range(len(ARGS)):
        start_time = time.time()
        print(ARGS[i])
        result = os.system(submit_format + ARGS[i])
        end_time = time.time()
        elapsed = int(end_time - start_time)
        print('elapsed : ' + str(elapsed))
        time.sleep(60*60 + 120 - elapsed)

