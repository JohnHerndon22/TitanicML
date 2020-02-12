import random

def reset(percent=80):
    rander = random.randrange(100)
    print(rander) 
    if rander < percent:
        return 1
    else:
        return 0

survived_cnt = 0
die_cnt = 0

for counter in range(100):
    if reset() == 1:
        survived_cnt +=1
    else:
        die_cnt +=1

print(survived_cnt, die_cnt)

# for counter in range(10):
#     print(random.randrange(100))
