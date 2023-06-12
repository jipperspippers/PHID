import numpy as np

class AddNoiseMixed(object):
    """add mixed noise to the given numpy array (B,H,W)
    Args:
        noise_bank: list of noise maker (e.g. AddNoiseImpulse)
        num_bands: list of number of band which is corrupted by each item in noise_bank"""
    def __init__(self, noise_bank, num_bands):
        assert len(noise_bank) == len(num_bands)
        self.noise_bank = noise_bank
        self.num_bands = num_bands

    def __call__(self, img):
        B, H, W = img.shape
        all_bands = np.random.permutation(range(B))
        pos = 0
        for noise_maker, num_band in zip(self.noise_bank, self.num_bands):
            if 0 < num_band <= 1:
                num_band = int(np.floor(num_band * B))
            bands = all_bands[pos:pos+num_band]
            pos += num_band
            img = noise_maker(img, bands)
        return img

class _AddNoiseNoniid(object):
    """add non-iid gaussian noise to the given numpy array (B,H,W)"""
    def __init__(self, sigmas):
        self.sigmas = np.array(sigmas) / 255.
    
    def __call__(self, img, bands):
        bwsigmas = self.sigmas[np.random.randint(0, len(self.sigmas), len(bands))]
        B, H, W = img.shape
        img = img + np.random.randn(*img.shape) * 3/255.
        for i, n in zip(bands,bwsigmas): 
            noise = np.random.randn(1,H,W)*n
            img[i,:,:] = img[i,:,:]+ noise
        return img 

class _AddStripeNoise(object):
    """add stripe noise to the given numpy array (B,H,W)"""
    def __init__(self, min_amount,max_amount,lam,var,direction):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount 
        self.direction = direction
        self.lam = lam   
        self.var = var   
    
    def __call__(self, img, bands):
        B, H, W = img.shape
        if self.direction == 'vertical':
            stype = H
        elif self.direction == 'horizontal':
            stype = W
        num_stripe = np.random.randint(np.floor(self.min_amount*stype), 
                                                    np.floor(self.max_amount*stype), len(bands))
        for i, n in zip(bands, num_stripe):
            loc = np.random.permutation(range(stype))
            loc = loc[:n]
            stripe = np.random.uniform(0,1, size=(len(loc),))*self.lam-self.var            
            if self.direction == 'vertical':
                sstripe = np.reshape(stripe, (-1, 1))
                img[i, loc, :] -= sstripe#np.reshape(stripe, (-1, 1))
            elif self.direction == 'horizontal':
                img[i, :, loc] -= np.reshape(stripe, (-1, 1))
        return img

class AddMixedNoiseW(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseNoniid([10,30,50,70]),
                           _AddStripeNoise(0.05, 0.35, 0.5, 0.25, 'horizontal')]
        self.num_bands = [1/3,1/3]

class AddMixedNoiseH(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseNoniid([10,30,50,70]),
                           _AddStripeNoise(0.05, 0.35, 0.5, 0.25, 'vertical')]
        self.num_bands = [2/3,1/3]

class AddStripeNoiseH(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseNoniid([0]),
                           _AddStripeNoise(0.05, 0.45, 0.7, 0.5, 'vertical')]
        self.num_bands = [0,2/3]

class AddStripeNoiseW(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseNoniid([0]),
                           _AddStripeNoise(0.05, 0.45, 0.7, 0.5, 'horizontal')]
        self.num_bands = [0,2/3]
