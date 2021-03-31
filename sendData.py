import sys
import time
qfrom Customer_Counter_main import Detector

start_time = 0
delay_time = 2
d = Detector()

while True:
    if start_time == 0:
        start_time = time.time()
    elapsed_time = time.time() - start_time
    if elapsed_time >= delay_time:
        print("hello")
        start_time = 0
    

