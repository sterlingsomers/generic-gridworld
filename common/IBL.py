import numpy as np
import random
# from autograd import numpy as np
# from autograd import elementwise_grad as egrad
# from autograd import grad
from sklearn.neighbors import KDTree
#TODO: instead of blending as weighted average use tensorflow for ONLY blending. So you assign a cross_entropy function
#TODO: have your data in memory ready for a placeholder and train! The problem will be the recall probabilities as these won't be of any use.
class IBL:
    def __init__(self, capacity, num_feats, num_actions, neighbors, temp):
        self.capacity = capacity
        self.num_feats = num_feats
        self.neighbors = neighbors
        self.num_actions = num_actions
        self.curr_capacity = 0
        self.curr_act_capacity = 0
        # self.memory = np.array([]) # First row is the timestep = 0. Going down we get more recent instances
        self.memory = np.empty((0,self.num_feats))#-np.ones([1,self.num_feats])
        #self.memo_actions = -np.ones([1,self.num_actions])
        self.memo_actions = np.empty([0,self.num_actions])
        self.activations = np.zeros([self.capacity]) # You need it a column vector as the indexing below doesnt work for a row vector
        self.tree = None
        self.similarity_f = self.relu
        self.timestep = 0
        self.temp = temp
        self.tm = 0#np.zeros([self.capacity]) # General time counter
        # NOTES: (below) least recently updated
        self.lru = np.zeros([self.capacity])# np.zeros([capacity, num_actions]) # it stores the tm that the
        # specific
        # instance, decision was
        # used. If it hasnt been used it remains 0 so it will be the least recently used.
        self.rng = np.random.RandomState(123456)

    def softmax(self, x): # Should be applied on a vector and NOT a matrix!
        """Compute softmax values for each sets of matching scores in x."""
        e_x = np.exp(x - np.max(x)) # You need a scaling param so 0s wont contribute that much and the small probas become bigger
        return e_x / e_x.sum(1).reshape(x.shape[0],1)

    def relu(self,x):
        x[x < 0] = 0
        return x

    def tanh(self,x):
        return np.tanh(x)

    # def add_instance_(self, instance):
    #     ''' Add an instance to the memory. If it is full remove the oldest one (top vector as we add below --most recent is the bottom).
    #     TODO: If it is a batch decision input we remove the NUMBER of entries from the top of the list.
    #     '''
    #     # Always it will get below for a vector (assuming that you use tables only at the beginning!)
    #     print('capacity',self.curr_capacity,'/', self.capacity)
    #     if instance.shape[0] - (self.capacity - self.curr_capacity) > 0:
    #         indx = instance.shape[0] - (self.capacity - self.curr_capacity)
    #         print('indx=',indx)
    #         self.memory[:-indx] = self.memory[indx:]  # We shift all instances up one timestep
    #         self.memory[-indx:] = instance
    #     # elif ((self.curr_capacity >= self.capacity )):
    #     #     indx = instance.shape[0]
    #     #     print('indx2=', indx)
    #     #     self.memory[:-indx] = self.memory[indx:]  # We shift all instances up one timestep
    #     #     self.memory[-indx:] = instance
    #     # if ((self.curr_capacity >= self.capacity )):
    #     #     self.memory[:-indx] = self.memory[indx:]  # We shift all instances up one timestep
    #     #     self.memory[-indx:] = instance
    #         # self.memory[:-1] = self.memory[1:]  # We shift all instances up one timestep
    #         # self.memory[-1] = instance # we assign the last slot to the new instance
    #         #self.memory = np.vstack((self.memory, instance))
    #
    #     else:
    #         #self.memory = np.append(self.memory, instance, axis=0) # Appends horizontal
    #         self.memory = np.vstack((self.memory, instance))
    #         self.curr_capacity = self.curr_capacity + instance.shape[0]
    #
    #     # self.tree = KDTree(self.states[:self.curr_capacity]) # till the current capacity (for pre-set static memory)
    #     self.tree = KDTree(self.memory, metric='manhattan') # choose num of trees!!!
    #     # if len(instance.shape)>1: # array ONLY IF WE ADD TO THE MEMORY WE INCREASE the capacity
    #     #     self.curr_capacity = self.curr_capacity + instance.shape[0]
    #         #rows = instance.shape[0]
    #     # else: # one column vector (n,) # might not need this if input is [array] and not array cauz from (n,)-->[n,1]
    #     #     self.curr_capacity = self.curr_capacity + 1
    #         #rows = 1

    def add_instance(self, instance): # THIS (4-Oct-2020)
        ''' Add a batch of instances to the memory. If it is full remove the oldest one (top vector as we add below
        --most recent is the bottom).
        TODO: If it is a batch decision input we remove the NUMBER of entries from the top of the list.
        '''
        # Always it will get below for a vector (assuming that you use tables only at the beginning!)
        # print('capacity',self.curr_capacity,'/', self.capacity)
        # NOTES: We just add and expand the memory
        self.memory = np.vstack((self.memory, instance))
        self.curr_capacity = self.curr_capacity + instance.shape[0]

        # NOTES: If the current capacity with the stuff we just added surpasses the memory pre-specified capacity then
        # we delete
        if self.curr_capacity - self.capacity > 0:
            indx = self.curr_capacity - self.capacity
            # print('indx=',indx)
            self.memory[:-indx] = self.memory[indx:]  # We shift all instances up one timestep
            # self.memory[-indx:] = instance
            rm = np.arange(self.curr_capacity, self.capacity, -1) - 1
            self.memory = np.delete(self.memory, rm, 0) # array, index, axis
            self.curr_capacity = self.memory.shape[0]
            # print('curr_capacity = ',self.curr_capacity)

        self.tm += 0.01  # update general timer of the memory (independent of which instances are getting updated)
        self.tree = KDTree(self.memory)#, metric='manhattan') # choose num of trees!!! REbuild the tree!


    def add(self, instances, values): # values are actions or expected return or decisions
        num_instances = instances.shape[0] # = how_many_to_add if capacity is not full!
        # self.curr_capacity = self.curr_capacity + num_instances
        if self.curr_capacity + num_instances > self.capacity: # If memo is full find the least recently used
            # instance and substitute its values with the new one. If you use >= then if max capacity is 10 and you
            # are at 8 and you need to add just 2 then old_index will be empty as how_many_to_delete=0. THERE WAS NO
            # ISSUE WITH EMPTY INDEX THOUGH!
            # find the LRU entry (key is the state projection to lower dims)
            how_many_to_delete = (self.curr_capacity + num_instances) - self.capacity # always > 0
            how_many_to_add = num_instances - how_many_to_delete # always >=0 # how many instances to add at the
            # bottom of the memory
            # Below: find the how_many_to_delete instances that minimize the lru (we do not check the whole lru
            # only the correct part of it.
            lru_min_index = np.argpartition(self.lru[:self.curr_capacity], how_many_to_delete)[:how_many_to_delete]
            # old_index = np.argmin(self.lru)
            self.memory[lru_min_index] = instances[:how_many_to_delete] # can we do this? indexed array assigning an array
            self.memo_actions[lru_min_index] = values[:how_many_to_delete]
            # Update timing of instances that just inserted in the position of others
            tms = self.tm * np.ones(how_many_to_delete)
            self.lru[lru_min_index] = tms

            if how_many_to_add > 0:
                tms_ = self.tm * np.ones(how_many_to_add)
                self.memory = np.vstack((self.memory, instances[-how_many_to_add:]))
                self.memo_actions = np.vstack((self.memo_actions, values[-how_many_to_add:]))
                self.lru[self.curr_capacity : self.curr_capacity + how_many_to_add] = tms_

            self.curr_capacity += how_many_to_add #num_instances

        else: # (MINE) Update and expand memory
            self.memory = np.vstack((self.memory, instances))
            self.memo_actions = np.vstack((self.memo_actions, values))
            tms = self.tm * np.ones(num_instances)
            self.lru[self.curr_capacity : self.curr_capacity + num_instances] = tms
            self.curr_capacity = self.memory.shape[0]
        # self.tm += 0.01 # update general timer of the memory (independent of which instances are getting updated)
        self.tree = KDTree(self.memory)#, metric='manhattan') # choose num of trees!!! Rebuild the tree!

    def add_action(self, decision): # THIS (4-Oct-2020)
        '''Add a decision/action to the memory. If it is full remove the oldest one (first element of the array). Instance should be a list.
        TODO: If it is a batch decision input we remove the number of entries from the top of the list.
        :param decision: a binary vector indicating which action was chosen
        '''
        self.memo_actions = np.vstack((self.memo_actions, decision))
        self.curr_act_capacity = self.curr_act_capacity + decision.shape[0]

        if self.curr_act_capacity - self.capacity > 0:
            indx = self.curr_act_capacity - self.capacity
            # print('indx=',indx)
            self.memo_actions[:-indx] = self.memo_actions[indx:]  # We shift all instances up one timestep
            # self.memo_actions[-indx:] = decision
            rm = np.arange(self.curr_act_capacity, self.capacity, -1) - 1
            self.memo_actions = np.delete(self.memo_actions, rm, 0)
            self.curr_act_capacity = self.memo_actions.shape[0]


    # def add_action_(self, decision):
    #     '''Add a decision/action to the memory. If it is full remove the oldest one (first element of the array). Instance should be a list.
    #     TODO: If it is a batch decision input we remove the number of entries from the top of the list.
    #     :param decision: a binary vector indicating which action was chosen
    #     '''
    #     if decision.shape[0] - (self.capacity - self.curr_act_capacity) > 0:
    #         indx = decision.shape[0] - (self.capacity - self.curr_act_capacity)
    #         print('indx=',indx)
    #         self.memo_actions[:-indx] = self.memo_actions[indx:]  # We shift all instances up one timestep
    #         self.memo_actions[-indx:] = decision
    #     # if self.curr_capacity + decision.shape[0] > self.capacity:
    #     #     indx = self.curr_capacity + decision.shape[0] - self.capacity
    #     # else:
    #     #     indx = decision.shape[0]
    #     # if ( (self.curr_act_capacity >= self.capacity) ): # or (self.curr_act_capacity == 0) ):
    #     #     # indx = decision.shape[0]
    #     #     self.memo_actions[:-indx] = self.memo_actions[indx:]  # We shift all instances up one timestep
    #     #     self.memo_actions[-indx:] = decision
    #     #
    #     #     # self.memo_actions[:-1] = self.memo_actions[1:]
    #     #     # self.memo_actions[-1] = decision
    #     #     #self.memo_actions = np.vstack((self.memo_actions, decision))
    #
    #     else:
    #         self.memo_actions = np.vstack((self.memo_actions, decision))
    #         self.curr_act_capacity = self.curr_act_capacity + decision.shape[0]
    #
    #     # if len(decision.shape)>1: # array
    #     #     self.curr_act_capacity = self.curr_act_capacity + decision.shape[0]
    #     # else: # one column vector (n,)
    #     #     self.curr_act_capacity = self.curr_act_capacity + 1

    def update_(self, s, a, r):
        # state = np.dot(self.matrix_projection, s.flatten()) # Dimensionality reduction with Random Projection (RP)-->21168 to 64
        # r is q_return
        self.peek(s, r, a, modify=True)
        # q_value = self.peek(state,r,a, modify = True) # Query the memory if q for s exists and retrieve it.
        # if q_value==None: # If none then two choices: Either subsittue the least recently used memory if memo is full or just expand memo and add new state
        #     self.ec_buffer[a].add(state,r)

    def peek_(self, instance, k):
        """ Find all the k instances that are most similar to the probe instance. """
        dist, ind = self.tree.query(instance, k=k) # Distance is by defualt eucleidean, you can change it, look sklearn doc
        # the distance for a=self.memory[ind[0]] is np.sqrt(np.sumnp.square(a-instance)))
        self.sub_memory = self.memory[ind] # indices are not sorted!!! So careful with everything --> means that the first dist is NOT the closest neighbor. Its just the k neighbors without particular order
        self.sub_memo_actions = self.memo_actions[ind]
        # self.sub_activations = self.activations[ind]
        #self.probs = self.softmax(-dist[0])
        return dist, ind # by using the tree we get the matching score without using the function below

    def update(self,instance, a, value):#, modify):
        #TODO: It could be done with ALL the experience and out of the reward loop
        # CAREFUL: Here we do not do knn!!! We search for the closest instances.
        # if self.curr_capacity==0:
        #     return None
        NACTIONS = self.num_actions
        actions = a
        returns = value
        # tree = KDTree(self.states[:self.curr_capacity])
        # print('query for the same instance in memory (k=1)')
        dist, ind = self.tree.query(instance, k=1) # Here you get a stored value (its used in estimate when q(a) != 0
        # ind are the indices in the MEMORY that the comparison took place (no matter the k) e.g. the closest inst in the memory with an incoming instance is the instance with ind 5 and the proximity (dist) is 9
        idxnon0 = np.where(dist != 0)  # Find identical instances in memory
        # Check if any of the queries exist in memory
        if 0 in dist:# NOTES: Some instances exist in memory (dist=0), so put
            idx0 = np.where(dist == 0)
            real_ind = ind[idx0]
            a = a.reshape(a.size, 1) # Reshape so indices can be used appropriately
            value = value.reshape(value.size, 1)

            self.lru[ind] = self.tm # only existing instances that are being updated are having their lru updated
            # BELOW is always True so you can remove it
            # if modify: # we replace or no (depending which one is bigger) the entry with the new new one. This happens when the entry exists in the memory

                # self.memo_actions[ind,a[idx0]] = max(self.q_values[ind],value) # According to Deepmind they replace the previous value with the new one if its bigger max(Qt,Rt) --> compare previous estimation with current return
            self.memo_actions[real_ind, a[idx0]] = np.maximum(self.memo_actions[real_ind, a[idx0]],
                                                                        value[idx0])
            #return self.q_values[ind]
        # You need a condition in case that all incoming instances exist in memory. If you return above though you wont be able to add
        # instances that do not exist (cases: all instances exist, mixed, all instances do not exist) NO! you dont
        # need cauz then idxnon0 will be empty so you won't have a problem
        if idxnon0: # NOTES: Now add ONLY the instances that are NOT in the memory: these are the instances that do not have a match in the memory so put their Return in as estimation
            # for these actions and add them into the memory. Also because it is mixed case, this case will search for
            value = returns.reshape(returns.size, 1)
            actions = actions.reshape(actions.size, 1)
            indx_batch = tuple([idxnon0[0]]) # Only reason for the tuple here is to avoid the warnings for indexing with non tuples
            instances = instance[indx_batch]
            # self.add_instance(instance[indx_batch]) # I use the 1st list of indices which indicates the index of
            # instances in the batch that are going to be added
            # Create the decisions array [batch x NACTIONS] that you will put in memory
            # We create (batch_ind, action_ind) pairs in order to be able to place the returns appropriately in a new array decisions which we will add in memo_actions
            batch_indices = np.arange(0, indx_batch[0].shape[0])
            indx_batch_cols = tuple([batch_indices, actions[idxnon0]]) # This is the same format that np.where creates
            # Below we create the [Q(a1),...,Q(aNACTIONS)] cauz state instance doesn't exist in memory. Only the
            # chosen actions will have their Qs updated with value
            decisions = -10*np.ones([indx_batch[0].shape[0], NACTIONS]) # decisions should have number of entries equal to the number of instances that NEED to get into the memory
            decisions[indx_batch_cols] = value[idxnon0] # put the values in
            # self.add_action(decisions)
            self.add(instances, decisions)
        # self.tm +=0.01

    def estimate(self,instance, knn):
        ''' Estimate does 2 things for EXPLOITATION phase (we need a Q in order to do a=argmaxQ(s,.) ):
        1. If an instance exists in memory, it retrieves any existing past decision.
        2. If an instance does not exist (εννοείται ότι οι decisions δεν υπάρχουν) then it uses knn function approximator
           in order to estimate Q
        :param instance: a batch of instances, one for every env

        Here, we create a decision array full of -1000. Then we replace any -1000 with any decision that exists in memory.
        Then we estimate via knn ALL the decisions in a different array and THEN we put the decisions array in the
        decisions that have still -1000.
        '''
        # if self.curr_capacity==0:
        #     return None
        NACTIONS = self.num_actions
        batch_size = instance.shape[0]
        decisions = -10 * np.ones([batch_size, NACTIONS])

        dist, ind = self.tree.query(instance, k=1) # Here you get a stored value (its used in estimate when q(a) != 0
        # ind are the indices in the MEMORY that the comparison took place (no matter the k) e.g. the closest inst in the memory with an incoming instance is the instance with ind 5 and the proximity (dist) is 9
        # idxnon0 = np.where(dist != 0)
        # Check if any of the queries exist in memory
        if 0 in dist:# basically it compares the key vector with the ONE closest neighbor to see if this entry exist
            # already
            # find the queries indx
            idx0 = np.where(dist == 0) # which incoming instances have entries in memory already
            real_ind = ind[idx0] # Find them in the main memory
            decisions[idx0[0]] = self.memo_actions[real_ind].copy() # Just retrieve the decisions as is. If there are
            #  -1000 they will be replaced below at the next if condition
            self.lru[real_ind] = self.tm # time of being used

        # NON EXISTENT ENTRIES ⟾ ESTIMATE THEM
        # NOTES: if there are decisions that do not have entries Q(,a) in the memory (i.e [s,Q(a1), -10,
        #  Q(a3)]), estimate these with knn. To do this we use matrix multiplication so we evaluate even the Q(a1)
        #  and Q(a2) BUT we do not use these as we take only the estimates with indices that of the non existent
        #  entries. We also use a default value to indicate empty entries (-10)
        dec_ind = np.where(decisions == -10)
        if dec_ind: # if there are any  empty ('-10') decisions (dec_ind is not empty)
            # print('query for estimate (k=knn)')
            dist, ind = self.peek_(instance, knn) # you might need ONLY one tree query --> YES you can do it with one query but the indexing code might not be working
            #TODO: activations = baselines[ind] + dist + noise, def baselines(lru,tm): ...
            probs = self.probabilities(-dist)
            # print('Blending')
            Q = self.blending(probs) # dimQ = [num_obs x NACTIONS]
            decisions[dec_ind] = Q[dec_ind]
        # a = 0.01 * np.ones(knn)
        # a[0] = 0
        # a = np.cumsum(a)
        # b = a + [self.tm] * knn
        # self.lru[ind[0]] = b
        # self.tm += 0.01 # Update timer

        # Now add ONLY the instances that are NOT in the memory: these are the instances that do not have a match in the memory
        # if idxnon0: # CASE: Some instances do not exist in memory
        #     actions = actions.reshape(actions.size, 1)
        #     indx_batch = tuple([idxnon0[0]]) # Only reason for the tuple here is to avoid the warnings for indexing with non tuples
        #     self.add_instance(instance[indx_batch]) # I use the 1st list of indices which indicate the incoming state order
        #     # Create the decisions array [batch x NACTIONS] that you will put in memory
        #     # We create (batch_ind, action_ind) pairs in order to be able to place the returns appropriately in a new array decisions which we will add in memo_actions
        #     batch_indices = np.arange(0, indx_batch[0].shape[0])
        #     indx_batch_cols = tuple([batch_indices, actions[idxnon0]]) # This is the same format that np.where creates
        #     decisions = -1000*np.ones([indx_batch[0].shape[0], NACTIONS]) # decisions should have number of entries equal to the number of instances that NEED to get into the memory
        #     decisions[indx_batch_cols] = value[idxnon0]
        return decisions

    def matching_score(self, instance):
        """ Compute Matching score for partial matching. """
        # Polynomial kernel: tanh(x.T*y + c)
        # Compute dot product between instance and selected memory entries
        similarity = self.sub_memory.dot(instance)
        # similarity = self.memory.dot(instance)
        x = similarity# + self.sub_activations # We use the transpose in order to sum 2 column vectors

        # return self.similarity_f(x)
        return x

    def probabilities(self, match_score):
        ''' Calculate retrieval probabilities for each instance '''
        return self.softmax(self.temp*match_score) # 5 seems the lowest value that produces the max acc. 2 for the larger dataset

    def blending(self, probabilities):#, slot):
        """ Weighted average implementation """
        # V = np.sum(self.sub_memo_actions[:,slot] * probabilities, axis=0)
        # V = np.matmul(probabilities, self.sub_memo_actions)
        # V = (probabilities.T * self.sub_memo_actions).sum(1)
        V = np.einsum('ijk,ij->ik', self.sub_memo_actions, probabilities) # Multi-query version
        # V = np.sum(self.memo_actions[:, slot] * probabilities, axis=0)
        # V = np.sum(np.array(self.sub_memo_actions*probabilities[:,None]),axis=0)
        # gradient_V = grad(V)
        # vec_gradient_V = egrad(V)
        return V#, gradient_V,vec_gradient_V

    def choose_action(self, obs, epsilon, knn, nenvs):
        self.tm += 0.01
        # EXPLORE
        if self.rng.random_sample() < epsilon:#random.random() < epsilon:
            # print('EXPLORE')
            return self.rng.choice(range(self.num_actions), nenvs)#np.random.randint(0, self.num_actions, nenvs) # do we need
            # num_actions-1?

        # dist, ind = self.peek_(obs, knn)
        # probs = self.probabilities(-dist)
        # Q = self.blending(probs) #
        # NOTES: MFEC original uses knn only when it encounters new states else it uses np.argmax. The previous
        #  states are are having their returns updated as in any tabular method!
        # EXPLOIT
        Q = self.estimate(obs,knn) # dimQ = [num_obs x NACTIONS]
            # NOTES: ↳ if memory exists then RETRIEVE IT, else use BLENDING
        # Tie breaking in random choice (good google search for numpy)
        maxes = Q.max(1) # for each row (=num_envs) we get the maximum
        a_max = []
        for i in range(maxes.size):
            a_maxt = np.random.choice(np.flatnonzero(Q[i,:] == Q[i,:].max())) # flatnonzero ignores Fasle (which is
            # equal to 0) and gets only the indices of the instances of the maximum value along the vector
            a_max.append(a_maxt)
        return np.array(a_max)#np.argmax(Q, axis=1)

    def saliency(self, point):
        """ Saliency calculation: Derivative of the blending wrt a feature
        :param: point: is the point in which we compute the derivative"""
        pass