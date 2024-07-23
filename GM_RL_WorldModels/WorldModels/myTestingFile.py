
'''
import numpy as np
import random

DIR_NAME = './data/rollout/'
episode_id = random.randint(0, 2**31 - 1)
filename = DIR_NAME + str(episode_id) + ".npz"

#File = open(filename, 'w')


np.savez_compressed(filename, [2,9.0,56]) 

#File.close()
'''

'''
from vae.arch import VAE

vae = VAE()

vae.set_weights('./vae/weights.h5')

ROLLOUT_DIR_NAME = "./data/rollout/"

rollout_data = np.load(ROLLOUT_DIR_NAME + '20942169.npz')

obs = rollout_data['obs']

print(vae.encoder.predict(obs))
'''
