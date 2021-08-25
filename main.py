import torch
import json
from scipy.optimize import linear_sum_assignment
import numpy as np
import random
import torch.optim as optim
from torch.autograd import Function
import matplotlib.pyplot as plt

manualSeed = 14030
beta1 = 0.5

# def secondaryLoss(qap, y):
#     output = torch.zeros(y.size(0))
#     for i in range(y.size(0)):
#         total = 0
#         for y in range(y.size(1)):
#             for x in range(y.size(2)):
#                 a = qap.facilities[y][x]
#                 first = torch.argmax(y[i][x])

def get_cost_function(qap):

    def loss(y):
        ret = torch.zeros(y.size(0))
        for i in range(y.size(0)):
            a = torch.mm(qap.facilities, y[i])
            b = torch.mm(a, torch.transpose(qap.distances, 0, 1))
            # b = torch.mm(a, qap.distances)
            c = torch.mm(b, torch.transpose(y[i], 0, 1))
            ret[i] = torch.trace(c) #/ (qap.diag_val * qap.diag_val) # this is differentiable
        return ret

    return loss

def row_norm_sink(mat):
    new_mat = mat.clone()
    for i in range(mat.shape[0]):
        new_mat[i] /= torch.sum(mat[i])

    return new_mat

def col_norm_sink(mat):
    new_mat = mat.clone()
    for j in range(mat.shape[1]):
        new_mat[:, j] /= torch.sum(mat[:, j])
    
    return new_mat

def sinkhorn_norm(mat, n):
    if n <= 0:
        return mat.clone()

    return col_norm_sink(row_norm_sink(sinkhorn_norm(mat, n-1))) 

def sink(mat, n=10):
    mat += 1E-3
    return sinkhorn_norm(mat, n)

def sink_row_2(mat, one_col, one_row):
    return mat / mat.mm(one_col).mm(one_row)
    

def sink_col_2(mat, one_col, one_row):
    return mat / one_col.mm(one_row).mm(mat)


def sink_2(mat, one_col, one_row, n=100):
    if n <= 0:
        return torch.exp(mat)

    return sink_col_2(sink_row_2(sink_2(mat, one_col, one_row, n-1), one_col, one_row), one_col, one_row)


class Sink2Functor:
    def __init__(self, one_col, one_row):
        self.one_col = one_col
        self.one_row = one_row

    def __call__(self, x, n):
        return sink_2(x, self.one_col, self.one_row, n)


class RowSinkLayer(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        output = x.clone()
        for i in range(x.size(0)):
            output[i] = row_norm_sink(output[i])
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        q = ctx.saved_tensors[0]
        grad_input = grad_output.clone()

        for i in range(q.size(0)):
            for j in range(q.size(1)):
                for k in range(q.size(2)):
                    derivative = 0
                    for y in range(q.size(1)):
                        curr = q[i][y][k]
                        a = 0
                        col_sum = torch.sum(q[i][:,k])

                        if y == j:
                            a = 1 / col_sum

                        b = grad_output[i][y][k] / (col_sum * col_sum)
                        derivative += curr * (a - b)

                    grad_input[i][j][k] = derivative
    
        return grad_input


class ColSinkLayer(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        output = x.clone()
        for i in range(x.size(0)):
            output[i] = col_norm_sink(output[i])
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        q = ctx.saved_tensors[0]
        grad_input = grad_output.clone()

        for i in range(q.size(0)):
            for j in range(q.size(1)):
                for k in range(q.size(2)):
                    derivative = 0
                    for x in range(q.size(2)):
                        curr = q[i][j][x]
                        a = 0
                        col_sum = torch.sum(q[i][x])

                        if x == k:
                            a = 1 / col_sum

                        b = grad_output[i][j][x] / (col_sum * col_sum)
                        derivative += curr * (a - b)

                    grad_input[i][j][k] = derivative
        
        return grad_input

class SinkLayer(torch.nn.Module):
    def __init__(self, iters):
        super(SinkLayer, self).__init__()
        self.iters = iters
        self.col_sink = ColSinkLayer.apply
        self.row_sink = RowSinkLayer.apply  

    def forward(self, x):
        output = x.clone() + 1E-3
        for i in range(self.iters):
            output = self.col_sink(self.row_sink(output))

        return output

class Model(torch.nn.Module):
    def __init__(self, input_features, n, iters=10):
        super(Model, self).__init__()

        odd = 2 * n
        self.add_odd = 1

        if n % 2 == 0:
            odd = 0
            self.add_odd = 0

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(input_features, n * 2),
            torch.nn.Linear(n * 2, n * n),
            torch.nn.Linear(n * n, 4 * n * n + odd)
        )


        # n = 2 * n - 8 + 2 * p
        p = (10 - n) // 2
        d = 1

        # 2 * n - 2 * d + 10 = n

        # print(p)
        if p < 0:
            p = 0
            d = (n - 8) // 2
 


        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(2 * n, 3 * n, 3, 1, p, d),
            torch.nn.Conv1d(3 * n, 2 * n, 3, 1, 0, 1),
            torch.nn.Conv1d(2 * n, n,     3, 1, 0, 1),
            torch.nn.Conv1d(n, n,         3, 1, 0, 1),
            torch.nn.Conv1d(n, n,         3, 1, 0, 1),
        )

        self.n = n
        self.sink = SinkLayer(iters)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 2 * self.n, 2 * self.n + self.add_odd)
        x = self.conv(x)
        # x = torch.abs(x)
        x = torch.pow(x, 2)
        x = self.sink(x)
        return x

                
        

def my_matching(matrix_batch):
  def hungarian(x):
    if x.ndim == 2:
      x = np.reshape(x, [1, x.shape[0], x.shape[1]])
    sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
    for i in range(x.shape[0]):
      sol[i, :] = linear_sum_assignment(-x[i, :])[1].astype(np.int32)
    return sol

  listperms = hungarian(matrix_batch.detach().cpu().numpy())
  listperms = torch.from_numpy(listperms)
  return listperms

class ProblemSpace:
    def __init__(self):
        pass

class QAP:
    def __init__(self, n):
        self.n = n
        self.diag_val = 0
        self.generate_problem(n)
        # self.load_problem()

    def generate_problem(self, n):
        self.facilities = 100 * torch.abs(torch.randn(n, n)) # x,y is the weight of 
        self.distances = 20 * torch.abs(torch.randn(n, n))

        self.facilities = torch.ceil(self.facilities)
        self.distances = torch.ceil(self.distances)

        for i in range(n):
            for j in range(i+1):
                if i == j:
                    self.facilities[i][j] = self.diag_val
                    self.distances[i][j] = self.diag_val
                else:
                    self.facilities[i][j] = self.facilities[j][i]
                    self.distances[i][j] = self.distances[j][i]

    def load_problem(self, filename_fac="fac.dat", filename_dist="dist.dat"):
        with open(filename_fac,"r") as f:
            loaded_list = eval(f.read())
        n = self.n#int(sqrt(len(loaded_list)))
        self.facilities = torch.tensor(loaded_list, dtype=torch.float32).view(n, n)

        with open(filename_dist,"r") as f:
            loaded_list = eval(f.read())
        # n = int(sqrt(len(loaded_list)))
        self.distances = torch.tensor(loaded_list, dtype=torch.float32).view(n, n)


    def __repr__(self):
        return "{{ \tn: {},\n\n\tfacilities: {},\n\n\tdistances: {} \n}}".format(self.n, self.facilities, self.distances)

def check_if_dsm(mat, eps=0.001):
    if mat.size(1) != mat.size(2):
        return False
    
    ret = []
    for i in range(mat.size(0)):
        result = True
        for y in range(mat.size(1)):
            row_sum = torch.sum(mat[i][y])
            if (abs(1 - row_sum) > eps):
                result = False
                break
        
        for x in range(mat.size(2)):
            col_sum = torch.sum(mat[i][:,x])
            if (abs(1 - col_sum) > eps):
                result = False
                break

        ret.append(result)
    
    return ret

def dsm_to_perm(dsm, T=False):
    match = my_matching(dsm)[0]
    out = torch.zeros(1, match.size(0), match.size(0))
    for i in range(match.size(0)):
        if T:
            out[0][i][match[i]] = 1
        else:
            out[0][match[i]][i] = 1
    return out


def test_sink():
#     # a = torch.arange(9).view(3, 3)
    a = torch.randn(1, 3, 3)
    a = torch.abs(a)
    # a = a * a
    # a = torch.pow(a, 2)
    print(a)
    a.requires_grad = True
#     a = torch.floor(a + 0.5)

#     b = torch.ones(a.shape[0]).view(3, 1)
#     c = torch.ones(a.shape[0]).view(1, 3)
#     # print(sink_2(a, b, c))
#     print(sink_2(a,b , c))
#     l = sink_2(a,b , c)
    # col_sink = ColSinkLayer.apply
    # row_sink = RowSinkLayer.apply
    # l = col_sink(row_sink(a))
    sinkLayer = SinkLayer(1000)
    l = sinkLayer(a)
    print(l)
    print(my_matching(l))
    print(check_if_dsm(l))
    perm = dsm_to_perm(l)

    inp_size = 100
    n = 50
    # network = Network(inp_size, n) # input doesnt really matter
    model = Model(inp_size, n)

    epochs = 10
    inpt = torch.randn(1, inp_size)
    qap = QAP(n)

    print(qap)

    loss = get_cost_function(qap)
    print(perm)
    score = loss(perm)
    print("score: {}".format(score))


def main():
    qap()
    # test_sink()

def image_to_scalar_rating():
    pass

def softmax_to_optimized_cost():
    pass

def mnist_confidence_max(): # can replace with any classification
    pass

def qap():
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    inp_size = 100
    n = 4

    model = Model(inp_size, n)

    epochs = 1000
    inpt = torch.randn(1, inp_size)
    qap = QAP(n)

    # print(qap)
    # exit()

    loss = get_cost_function(qap)

    # optimizer = optim.Adam(model.parameters(), lr=0.07, betas=(0.9, 0.999))
    # optimizer = optim.SGD(model.parameters(), lr=0.1)
    optimizer = optim.Adagrad(model.parameters(), lr=0.4)

    scores = []
    best_score = 1E10
    best_dsm = None

    for i in range(epochs):
        model.zero_grad()
        dsm = model(inpt)
        score = loss(dsm)
        score.backward()
        optimizer.step()

        scores.append(score.mean().item())

        if score < best_score:
            best_score = score
            best_dsm = dsm.clone()


        print("[{}/{}] Score: {}, DSM: {}".format(i, epochs, score.mean().item(), check_if_dsm(dsm)))

    


    perm_t = dsm_to_perm(best_dsm, True)
    perm_nt = dsm_to_perm(best_dsm)

    print(best_dsm)
    print(perm_t)

    print("score t: {}".format(loss(perm_t)))
    print("score nt: {}".format(loss(perm_nt)))
    
    plt.figure(figsize=(10,5))
    plt.title("Scores Over time")
    plt.plot(scores,label="Scores")
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()