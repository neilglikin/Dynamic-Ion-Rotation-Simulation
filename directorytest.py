import os 
import numpy as np

x = [1,2,3,4]
filename = 'test'
savepath_string = os.path.dirname(os.path.realpath(__file__))
savestring = savepath_string + '/testfolder'
if not os.path.isdir(savestring):
    os.mkdir(savestring)
savestring_file = savestring + '/test3'
np.save(savestring_file, x)
