import numpy as np
import copy

def gradients(prev_weights, new_weights):
    """
    gradients here should be added to the model, unlike the its conventional mathematical definition
    """
    gradients = []
    for i in range(len(prev_weights)):
        gradients.append(new_weights[i] - prev_weights[i])
    return gradients

def m_gradients(prev_weights, new_weights):
    """
    mathmatical gradients, which should be subtracted from the model params
    """
    gradients = []
    for i in range(len(prev_weights)):
        gradients.append(prev_weights[i] - new_weights[i])
    return gradients

def add_weights(w1, w2):
    if w1 == None:
        return w2
    res = []
    for i in range(len(w1)):
        res.append(w1[i] + w2[i])
    return res

def multiply_weights(w, num):
    res = []
    for i in range(len(w)):
        res.append(w[i] * num)
    return res

def avg_weights(my_weights, other_weights):
    if my_weights == None:
        return other_weights
    weights = [my_weights, other_weights]
    agg_weights = list()
    coeff = 0.5
    for weights_list_tuple in zip(*weights):
        agg_weights.append(np.array([np.average(np.array(w), axis=0, weights=[coeff, 1.-coeff]) for w in zip(*weights_list_tuple)]))
    
    return agg_weights

def enlarge_weights(target, source):
    n = len(target)
    if n != len(source):
        raise ValueError("The number of layers do not match!")
    for i in range(n):
        rep_put_weights(target[i], source[i])

# adaptively put repeated weights of source to target
def rep_put_weights(target, source):
    if len(target.shape) == 2:
        tr, tc = target.shape
        sr, sc = source.shape
        if tr != sr:
            if tc != sc:
                rep_put_diag(target, source)
            else:
                rep_put_2d(target, source)
        else:
            if tc != sc:
                rep_put_2d(target, source)
            else:
                target[:] = source
    elif len(target.shape) == 1:
        rep_put_1d(target, source)

def rep_put_1d(target, arr):
    tn = target.shape[0]
    n = arr.shape[0]
    for i in range(int(tn/n)):
        for j in range(n):
            target[j+i*n] = arr[j]

# repeatedlyh puts arr to target along one axis
def rep_put_2d(target, arr):
    tn, tm = target.shape
    n, m = arr.shape
    if tn != n:
        for i in range(int(tn/n)):
            put(target, i*n, 0, arr)
    elif tm != m:
        for i in range(int(tm/m)):
            put(target, 0, i*m, arr)
        
# repeatedly puts arr to target diagonally
def rep_put_diag(target, arr):
    tn, tm = target.shape
    n, m = arr.shape
    for i in range(int(tn/n)):
        put(target, n*i, m*i, arr)

def put(target, row, col, arr):
    n, m = arr.shape
    for i in range(n):
        for j in range(m):
            target[row+i][col+j] = arr[i][j]

class SelectWeightsAdv():
    def __init__(self):
        self.masks = []
        self.count = []

    def get_probs(self, target):
        # get probability for selecting weights
        cnt_sum = np.sum(target, axis=0)
        probs = (np.max(cnt_sum) - cnt_sum) + 0.01
        probs /= np.sum(probs)
        return probs
            
    def select_weights(self, target, select): 
        self.select = copy.deepcopy(select)
        self.target = copy.deepcopy(target)
        
        # if len(self.count) == 0:
        #     for w in target:
        #         self.count.append(np.zeros(w.shape))
        # reset mask
        self.masks = []
        
        n = len(self.target)
        if n != len(self.select):
            raise ValueError("The number of layers do not match!")
        for i in range(n):
            mask = np.zeros(self.target[i].shape, dtype='bool')
            
            if (i == 0): # input layer 
                probs = self.get_probs(self.target[i])
                cols = np.random.choice(np.arange(self.target[i].shape[1]), 
                                        size=self.select[i].shape[1], replace=False)
                for r in np.arange(self.target[i].shape[0]):
                    for c in cols:
                        mask[r][c] = True
                self.select[i] = self.target[i][mask].reshape(self.select[i].shape)
                
            elif i == n-2: # output weights
                rows = cols
                for r in rows:
                    for c in np.arange(self.target[i].shape[1]):
                        mask[r][c] = True
                self.select[i] = self.target[i][mask].reshape(self.select[i].shape)
            elif i == n-1: # output bias
                mask |= True
                
            elif len(self.target[i].shape) == 2: # weights
                rows = cols
                probs = self.get_probs(self.target[i])
                
                cols = np.random.choice(np.arange(self.target[i].shape[1]), 
                                        size=self.select[i].shape[1],replace=False)
                for r in rows:
                    for c in cols:
                        mask[r][c] = True
                self.select[i] = self.target[i][mask].reshape(self.select[i].shape)
                
            elif len(self.target[i].shape) == 1: # bias
                for c in cols:
                    mask[c] = True
                self.select[i] = self.target[i][mask]
            
            self.masks.append(mask)
        # for i in range(len(self.masks)):
        #     self.count[i] += self.masks[i]
        
        return self.select
    
    def get_selected(self):
        return self.select

    def update_target(self, select):
        for i in range(len(select)):
            self.target[i][self.masks[i]] = select[i].ravel()
        return self.target
    
    def get_target_from_selected(self, weights):
        res = []
        for i in range(len(weights)):
            w = np.zeros(shape=self.target[i].shape)
            w[self.masks[i]] = weights[i].ravel()
            res.append(w)
        return res
    
    def get_selected_adam_optimizer_weights(self, weights):
        res = []
        res.append(weights[0]) # iter num
        for i in range(1, len(weights)):
            if i <= len(self.masks):
                res.append(weights[i][self.masks[i-1]].reshape(self.select[i-1].shape))
            else:
                res.append(weights[i][self.masks[i-len(self.masks)-1]].reshape(self.select[i-len(self.masks)-1].shape))
        return res
    
    def get_target_adam_optimizer_weights(self, weights):
        res = []
        res.append(weights[0]) # iter num
        for i in range(1, len(weights)):
            if i <= len(self.masks):
                idx = i-1
            else:
                idx = i-len(self.masks)-1
            w = np.zeros(shape=self.target[idx].shape)
            w[self.masks[idx]] = weights[i].ravel()
            res.append(w)
        return res

class MomentumSelectWeights(SelectWeightsAdv):
    def __init__(self):
        super().__init__()
    
    def get_target_from_selected_w_momentum(self, momentum, weights):
        res = []
        for i in range(len(weights)):
            w = copy.deepcopy(momentum[i])
            w[self.masks[i]] = weights[i].ravel()
            res.append(w)
        return res

    def get_target_from_selected_w_momentum_avg(self, momentum, weights):
        res = []
        for i in range(len(weights)):
            w = copy.deepcopy(momentum[i])
            w[self.masks[i]] = (w[self.masks[i]] + weights[i].ravel()) / 2
            res.append(w)
        return res


class SelectWeightsNoWeighting(SelectWeightsAdv):
    def __init__(self):
        super().__init__()
    
    def get_probs(self, target):
        probs = np.ones(target.shape[1])
        probs /= np.sum(probs)
        return probs

class SelectWeightsConv(SelectWeightsAdv):
    def __init__(self):
        super().__init__()

    def select_weights(self, target, select):
        self.masks = []
        self.select = copy.deepcopy(select)
        self.target = target

        n = len(self.target)
        if n != len(select):
            print('target layers is {} while select is {}'.format(n, len(select)))
            for w in self.target:
                print(w.shape)
            for w in select:
                print(w.shape)
            raise ValueError("The number of layers do not match!")
        for i in range(n):
            mask = np.zeros(self.target[i].shape, dtype='bool')
            
            if i == 8: # flatten layer   
                cols = np.random.choice(np.arange(self.target[i].shape[1]), size=select[i].shape[1], replace=False)
                depth = self.target[i-1].shape[0]
                for f in filters:
                    for c in cols:
                        for ii in range(6):
                            for j in range(6):
                                mask[ii * 6 * depth + j * depth + f][c] = True
                masked = self.target[i][mask]
                self.select[i] = masked.reshape(select[i].shape)
                
            elif i == 9: # flatten layer bias
                for c in cols:
                    mask[c] = True
                self.select[i] = self.target[i][mask]
                
            elif i == 10: # dense layer
                rows = cols
                for r in rows:
                    for c in np.arange(self.target[i].shape[1]):
                        mask[r][c] = True
                self.select[i] = self.target[i][mask].reshape(select[i].shape)
                
            elif i == n-1: # output bias
                mask |= True
                
            elif i == 0: # first conv weights
                filters = np.random.choice(np.arange(self.target[i].shape[3]), size=select[i].shape[3], replace=False)
                for f in filters:
                    mask[:,:,:,f] = True
                self.select[i] = self.target[i][mask].reshape(select[i].shape)
                
            elif len(self.target[i].shape) == 4: # conv weights
                prev_filters = filters
                filters = np.random.choice(np.arange(self.target[i].shape[3]), size=select[i].shape[3], replace=False)
                for f in filters:
                    for p in prev_filters:
                        mask[:,:,p,f] = True
                self.select[i] = self.target[i][mask].reshape(select[i].shape)
                
            elif len(self.target[i].shape) == 1: # bias
                for f in filters:
                    mask[f] = True
                self.select[i] = self.target[i][mask]
            
            self.masks.append(mask)
        return self.select

class MomentumSelectWeightsConv(SelectWeightsConv):
    def __init__(self):
        super().__init__()
    
    def get_target_from_selected_w_momentum(self, momentum, weights):
        res = []
        for i in range(len(weights)):
            w = copy.deepcopy(momentum[i])
            w[self.masks[i]] = weights[i].ravel()
            res.append(w)
        return res

    def get_target_from_selected_w_momentum_avg(self, momentum, weights):
        res = []
        for i in range(len(weights)):
            w = copy.deepcopy(momentum[i])
            w[self.masks[i]] = (w[self.masks[i]] + weights[i].ravel()) / 2
            res.append(w)
        return res

# adaptively select weights of target
def select_weights(target, select):
    n = len(target)
    if n != len(select):
        raise ValueError("The number of layers do not match!")
    for i in range(n):   
        if len(target[i].shape) == 2:
            select[i], _ = select_2d(target[i], select[i])
        elif len(target[i].shape) == 1:
            select[i], _ =  select_1d(target[i], select[i])

def select_2d(target, select):
    mask = np.zeros(target.shape, dtype='bool')
    rows = np.random.choice(np.arange(target.shape[0]), size=select.shape[0], replace=False)
    cols = np.random.choice(np.arange(target.shape[1]), size=select.shape[1], replace=False)
    for i in rows:
        for j in cols:
            mask[i][j] = True
    return target[mask].reshape(select.shape), mask

def select_1d(target, select):
    mask = np.zeros(target.shape[0], dtype='bool')
    cols = np.random.choice(np.arange(target.shape[0]), size=select.shape[0], replace=False)
    for i in cols:
        mask[i] = True
    return target[mask], mask

######
# for quantization
# https://github.com/sunjunaimer/LAQ/blob/master/NeurIPS2019:LAQ/LAQ.py


def vec_to_grad(vec, grads):
    le = len(grads)
    new_grad = []
    cur = 0
    for i in range(0, le):
        s_sum = 1
        for s in grads[i].shape:
            s_sum *= s
        new_grad.append(vec[cur:cur+s_sum].reshape(grads[i].shape))
        cur += s_sum
    return new_grad

def grad_to_vec(grads):
    vec = np.array([])
    g_numpy = [g.numpy() for g in grads]
    for g in g_numpy:
        vec = np.concatenate((vec, g.ravel()), axis=0)
    return vec

def weights_to_vec(weights):
    vec = np.array([])
    w_numpy = [w for w in weights]
    for w in weights:
        vec = np.concatenate((vec, w.ravel()), axis=0)
    return vec


# SL: quantize gradients (vec) based on prior quantized gradients (v2) according to the number of bits (b)
def quant_d(vec,v2,b):
    n=len(vec)
    r=max(abs(vec-v2))
    delta=r/(np.floor(2**b)-1)
    quantv=v2-r+2*delta*np.floor((vec-v2+r+delta)/(2*delta))
    return quantv

def quant(vec, b):
    n = len(vec)
    max_val = max(abs(vec))
    tau = max_val / (np.floor(2**b)-1)
    quantv = 2 * tau * np.floor((vec + max_val + tau)/(2 * tau)) - max_val
    return quantv
