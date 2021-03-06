import numpy as np




maps = {'pacman':{'colors':{0:'gray',1:'black',2:'white',3:'yellow',4:'pink',5:'aqua'},
                  'properties':{2:{'moveTo':'moveToPill','free':True}, 3:{'moveTo':'moveToPill'},4:{'moveTo':'moveToPill'},5:{'moveTo':'moveToPill'}},
                  'free-spaces':[0,2],
                  'map':np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                                 [1,2,2,2,2,2,4,2,2,2,2,2,2,1,1,2,2,2,2,2,2,4,2,2,2,2,2,1],
                                 [1,2,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2,1],
                                 [1,3,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,3,1],
                                 [1,2,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2,1],
                                 [1,4,2,2,2,2,4,2,2,4,2,2,4,2,2,4,2,2,4,2,2,4,2,2,2,2,4,1],
                                 [1,2,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,2,1],
                                 [1,2,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,2,1],
                                 [1,2,2,2,2,2,4,1,1,2,2,2,2,1,1,2,2,2,2,1,1,4,2,2,2,2,2,1],
                                 [1,1,1,1,1,1,2,1,1,1,1,1,0,1,1,0,1,1,1,1,1,2,1,1,1,1,1,1],
                                 [1,1,1,1,1,1,2,1,1,1,1,1,0,1,1,0,1,1,1,1,1,2,1,1,1,1,1,1],
                                 [1,1,1,1,1,1,2,1,1,0,0,0,5,0,0,5,0,0,0,1,1,2,1,1,1,1,1,1],
                                 [1,1,1,1,1,1,2,1,1,0,1,1,1,0,0,1,1,1,0,1,1,2,1,1,1,1,1,1],
                                 [1,1,1,1,1,1,2,1,1,0,1,0,0,0,0,0,0,1,0,1,1,2,1,1,1,1,1,1],
                                 [0,0,0,0,0,0,4,0,0,4,1,0,0,0,0,0,0,1,4,0,0,4,0,0,0,0,0,0],
                                 [1,1,1,1,1,1,2,1,1,0,1,0,0,0,0,0,0,1,0,1,1,2,1,1,1,1,1,1],
                                 [1,1,1,1,1,1,2,1,1,0,1,1,1,1,1,1,1,1,0,1,1,2,1,1,1,1,1,1],
                                 [1,1,1,1,1,1,2,1,1,0,0,0,0,0,0,0,0,0,0,1,1,2,1,1,1,1,1,1],
                                 [1,1,1,1,1,1,2,1,1,0,1,1,1,1,1,1,1,1,0,1,1,2,1,1,1,1,1,1],
                                 [1,1,1,1,1,1,2,1,1,0,1,1,1,1,1,1,1,1,0,1,1,2,1,1,1,1,1,1],
                                 [1,2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,1],
                                 [1,2,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2,1],
                                 [1,2,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2,1],
                                 [1,3,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,2,3,1],
                                 [1,1,1,2,1,1,2,1,1,2,1,1,1,1,1,1,1,1,2,1,1,2,1,1,2,1,1,1],
                                 [1,1,1,2,1,1,2,1,1,2,1,1,1,1,1,1,1,1,2,1,1,2,1,1,2,1,1,1],
                                 [1,2,2,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,2,2,1],
                                 [1,2,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,2,1],
                                 [1,2,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,2,1],
                                 [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],
                                 [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])},
'pacman-default':{'colors':{0:'gray',1:'black',2:'gray',3:'gray',4:'gray',5:'gray'},
                  'properties':{2:{'doorway':True,'moveTo':'moveToDefault'},3:{'doorway':True,'moveTo':'moveToDefault'},4:{'doorway':True,'moveTo':'moveToDefault'},5:{'doorway':True,'moveTo':'moveToDefault'}},
                  'free-spaces':[2,3,4,5],
                  'map':np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                                 [1,2,2,2,2,2,4,2,2,2,2,2,2,1,1,2,2,2,2,2,2,4,2,2,2,2,2,1],
                                 [1,2,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2,1],
                                 [1,3,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,3,1],
                                 [1,2,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2,1],
                                 [1,4,2,2,2,2,4,2,2,4,2,2,4,2,2,4,2,2,4,2,2,4,2,2,2,2,4,1],
                                 [1,2,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,2,1],
                                 [1,2,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,2,1],
                                 [1,2,2,2,2,2,4,1,1,2,2,2,2,1,1,2,2,2,2,1,1,4,2,2,2,2,2,1],
                                 [1,1,1,1,1,1,2,1,1,1,1,1,0,1,1,0,1,1,1,1,1,2,1,1,1,1,1,1],
                                 [1,1,1,1,1,1,2,1,1,1,1,1,0,1,1,0,1,1,1,1,1,2,1,1,1,1,1,1],
                                 [1,1,1,1,1,1,2,1,1,0,0,0,5,0,0,5,0,0,0,1,1,2,1,1,1,1,1,1],
                                 [1,1,1,1,1,1,2,1,1,0,1,1,1,0,0,1,1,1,0,1,1,2,1,1,1,1,1,1],
                                 [1,1,1,1,1,1,2,1,1,0,1,0,0,0,0,0,0,1,0,1,1,2,1,1,1,1,1,1],
                                 [0,0,0,0,0,0,4,0,0,4,1,0,0,0,0,0,0,1,4,0,0,4,0,0,0,0,0,0],
                                 [1,1,1,1,1,1,2,1,1,0,1,0,0,0,0,0,0,1,0,1,1,2,1,1,1,1,1,1],
                                 [1,1,1,1,1,1,2,1,1,0,1,1,1,1,1,1,1,1,0,1,1,2,1,1,1,1,1,1],
                                 [1,1,1,1,1,1,2,1,1,0,0,0,0,0,0,0,0,0,0,1,1,2,1,1,1,1,1,1],
                                 [1,1,1,1,1,1,2,1,1,0,1,1,1,1,1,1,1,1,0,1,1,2,1,1,1,1,1,1],
                                 [1,1,1,1,1,1,2,1,1,0,1,1,1,1,1,1,1,1,0,1,1,2,1,1,1,1,1,1],
                                 [1,2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,1],
                                 [1,2,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2,1],
                                 [1,2,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2,1],
                                 [1,3,2,2,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,2,2,3,1],
                                 [1,1,1,2,1,1,2,1,1,2,1,1,1,1,1,1,1,1,2,1,1,2,1,1,2,1,1,1],
                                 [1,1,1,2,1,1,2,1,1,2,1,1,1,1,1,1,1,1,2,1,1,2,1,1,2,1,1,1],
                                 [1,2,2,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,2,2,2,1],
                                 [1,2,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,2,1],
                                 [1,2,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,2,1],
                                 [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1],
                                 [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])},

        'house': {'colors':{0:'gray',1:'black',2:'gray'},
                  'properties':{2:{'doorway':True,'moveTo':'moveToDefault'}},
                  'free-spaces':[0,2],
                  'permanent':[1,2,0],
                  'map': np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [1, 1, 1, 2, 1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                                   [1, 1, 1, 2, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                                   [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1],
                                   [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                                   [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                                   [1, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1],
                                   [1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])},
        'small-empty': {'colors':{0:'gray'},
                        'free-spaces':[0],
                        'permanent':[0],
                        'map': np.array([[1,1,1,1,1,1,1,1,1,1],
                                         [1,0,0,0,0,0,0,0,0,1],
                                         [1,0,0,0,0,0,0,0,0,1],
                                         [1,0,0,0,0,0,0,0,0,1],
                                         [1,0,0,0,0,0,0,0,0,1],
                                         [1,0,0,0,0,0,0,0,0,1],
                                         [1,0,0,0,0,0,0,0,0,1],
                                         [1,0,0,0,0,0,0,0,0,1],
                                         [1,0,0,0,0,0,0,0,0,1],
                                         [1,1,1,1,1,1,1,1,1,1]])},
        'small-portals': {'colors':{0:'gray'},
                        'free-spaces':[0],
                        'permanent':[0],
                        'map': np.array([[1,1,1,1,0,0,1,1,1,1],
                                         [1,0,0,0,0,0,0,0,0,1],
                                         [1,0,1,0,1,1,0,1,0,1],
                                         [1,0,0,0,0,0,0,1,0,1],
                                         [0,0,1,0,1,1,0,1,0,0],
                                         [0,0,1,0,1,1,0,0,0,0],
                                         [1,0,1,0,0,0,0,1,0,1],
                                         [1,0,1,1,1,1,0,1,0,1],
                                         [1,0,0,0,0,0,0,0,0,1],
                                         [1,1,1,1,0,0,1,1,1,1]])}
        }
