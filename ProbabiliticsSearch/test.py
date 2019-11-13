import random
import numpy as np

terrain = ["flat", "hilly", "forested", "caves"]
terrain.remove("hilly")
print(terrain)
print(random.choice(terrain))
for i in range(4):
    print(terrain[i])