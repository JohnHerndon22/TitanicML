# random.py
import random

def determine_life(p):
    return int(random.random() <= p)

livecnt = 0
deadcnt = 0
for x in range(1000):
    if determine_life(.384) == 1: 
        livecnt+=1
        print('passenger lives! ')
    else:
        deadcnt+=1
        print('passenger dies! ')

print(str(livecnt)+ ' made it')
print(str(deadcnt)+ ' at the bottom of the sea!')