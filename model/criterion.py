import torch
import torch.nn as nn
import numpy as np



### compute MMD and  CMMD scores within the framework of Pytorch.
### we adopt Gaussian kernel with band width set to median pairwise squared distances.
# on the training data
class MMD_loss(nn.Module):
    def __init__(self, kernel_type='mmd', kernel_mul=2.0, kernel_num=5,eplison=0.001):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

        self.eplison = eplison

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        '''
        Source:[sample_len1, features_len]
        Target:[sample_len2, features_len]
        returns: [sample1, sample2] guassian kernel gram matrices.  
        '''
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val) / self.kernel_num

    def linear2_mmd(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def rbf_mmd(self,source,target):
        '''
        Return MMD score based on guassian kernel. 
        '''
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(
            source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        # with torch.no_grad():
        XX = torch.mean(kernels[:batch_size, :batch_size])

        YY = torch.mean(kernels[batch_size:, batch_size:])
        XY = torch.mean(kernels[:batch_size, batch_size:])
        YX = torch.mean(kernels[batch_size:, :batch_size])
        loss = XX + YY - XY - YX
        # torch.cuda.empty_cache()
        return torch.sqrt(loss)

    def rbf_class_mmd(self,sX,tX,sY,tY):
        '''
        Return the sum of all MMDs of different classes. (Non differentiable.)
        '''

        n_sample1 = sX.size(0)
        n_sample2 = tX.size(0)
        device = sX.device
        batch_size = sX.size(0)

        class_mmd = 0.0

        label_num = torch.max(sY) + 1

        for current_label in range(label_num):
            current_sX = sX[sY==current_label]
            current_tX = tX[tY==current_label]
            if current_sX.size(0) == 0:
                current_sX = torch.zeros(5, sX.size(1)).to(device)
            if current_tX.size(0) == 0:
                current_tX = torch.zeros(5, tX.size(1)).to(device)

            class_mmd += self.rbf_mmd(current_sX,current_tX) 

        return class_mmd


    def rbf_cmmd(self,sX,tX,sY,tY):
        '''
        Return CMMD score based on guassian kernel. 
        '''
        n_sample1 = sX.size(0)
        n_sample2 = tX.size(0)
        device = sX.device
        batch_size = sX.size(0)
        xkernels = self.guassian_kernel(sX,tX, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,  fix_sigma=self.fix_sigma)
        ykernels = self.guassian_kernel(sY,tY, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)

        X11 = xkernels[:batch_size,:batch_size]
        X21 = xkernels[batch_size:,:batch_size]
        X22 = xkernels[batch_size:,batch_size:]

        Y11 = ykernels[:batch_size,:batch_size]
        Y12 = ykernels[:batch_size,batch_size:]
        Y22 = ykernels[batch_size:,batch_size:]
        X11_inver = torch.inverse(X11 + self.eplison * n_sample1* torch.eye(n_sample1).to(device))
        X22_inver = torch.inverse(X22 + self.eplison * n_sample2* torch.eye(n_sample2).to(device))

        cmmd1 = -2.0/(n_sample1*n_sample2)*torch.trace(X21.mm(X11_inver).mm(Y12).mm(X22_inver))
        cmmd2 = 1.0/(n_sample1*n_sample1)*torch.trace(Y11.mm(X11_inver))
        cmmd3 =1.0/(n_sample2*n_sample2)*torch.trace(Y22.mm(X22_inver))

        loss = cmmd1 + cmmd2 + cmmd3
        return torch.sqrt(loss)

    def forward(self, source, target,sourceY=None, targetY=None):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'mmd':
            return self.rbf_mmd(source,target)
        elif self.kernel_type== 'cmmd':
            return self.rbf_cmmd(source,target,sourceY,targetY)
        
            

if __name__ == "__main__":
    ### test. 
    a = np.arange(1, 10).reshape(3, 3)
    b = [[7, 6, 5], [4, 3, 2], [1, 1, 8], [0, 2, 5]]
    b = np.array(b)
    na = torch.FloatTensor(a)
    nb = torch.FloatTensor(b)

    ay = torch.FloatTensor([[.6,.4],[.8,.9],[.1,.2]])
    by = torch.FloatTensor([[0.1,0.9],[0,1],[1,0],[0.5,0.5]])

    mmd_c = MMD_loss(kernel_type='mmd', kernel_mul=2.0, kernel_num=5)
    cmmd_c = MMD_loss(kernel_type='cmmd', kernel_mul=2.0, kernel_num=5,eplison=0.0000001)

    print(mmd_c(na,nb))
    print(cmmd_c(na,nb,ay,by))