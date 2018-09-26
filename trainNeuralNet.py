import mxnet as mx
from mxnet import nd, autograd


######some functions#####

def adam(params, vs, sqrs, lr, batch_size, t):
    beta1 = 0.9
    beta2 = 0.999
    eps_stable = 1e-8

    for param, v, sqr in zip(params, vs, sqrs):
        g = param.grad / batch_size

        v[:] = beta1 * v + (1. - beta1) * g
        sqr[:] = beta2 * sqr + (1. - beta2) * nd.square(g)

        v_bias_corr = v / (1. - beta1 ** t)
        sqr_bias_corr = sqr / (1. - beta2 ** t)

        div = lr * v_bias_corr / (nd.sqrt(sqr_bias_corr) + eps_stable)
        param[:] = param - div

def SE(yhat, y):
    return nd.sum((yhat - y) ** 2)

def net(x,params):
    L2 = nd.sigmoid(nd.dot(x,params[0])+params[1])
    L3 = nd.sigmoid(nd.dot(L2,params[2])+params[3])
    L4 = nd.sigmoid(nd.dot(L3,params[4])+params[5])
    L5 = nd.sigmoid(nd.dot(L4,params[6])+params[7])
    Lout = nd.sigmoid(nd.dot(L5,params[8])+params[9])
    return Lout

######settings######
myCtx = mx.cpu(0)

epochs=10000
nbIter=50

batch_size= 100


######Hamming code#####
k=4
n=7 #8
lr=0.001

G= nd.array([[1, 0, 0, 0,1,0,1],
             [0, 1, 0, 0,1,1,0],
             [0, 0, 1, 0,1,1,1],
             [0, 0, 0, 1,0,1,1]],ctx=myCtx)

Gbis= nd.array([[1, 0, 0, 0,1,0,1,1],
                [0, 1, 0, 0,1,1,0,1],
                [0, 0, 1, 0,1,1,1,0],
                [0, 0, 0, 1,0,1,1,1]],ctx=myCtx)

t=1 #adam variable
vs = []
sqrs = []

########params######

#nd Neurons per Layer
nbL2 = 7
nbL3 = 7
nbL4 = 7
nbL5 = 7

W1 = nd.random_normal(shape=(n,nbL2),ctx=myCtx)
b1 = nd.random.normal(shape = (nbL2,),ctx=myCtx)

W2 = nd.random_normal(shape=(nbL2,nbL3),ctx=myCtx)
b2 = nd.random.normal(shape = (nbL3,),ctx=myCtx)

W3 = nd.random_normal(shape=(nbL3,nbL4),ctx=myCtx)
b3 = nd.random.normal(shape = (nbL4,),ctx=myCtx)

W4 = nd.random_normal(shape=(nbL4,nbL5),ctx=myCtx)
b4 = nd.random.normal(shape = (nbL5,),ctx=myCtx)

W5 = nd.random_normal(shape=(nbL5,k),ctx=myCtx)
b5 = nd.random.normal(shape = (k,),ctx=myCtx)

params = [W1,b1,W2,b2,W3,b3,W4,b4,W5,b5]

for param in params:
    param.attach_grad()
    vs.append(param.zeros_like())
    sqrs.append(param.zeros_like())

######Main loop ########

for i in range(epochs):
    efficiency = 0
    cumuLoss = 0
    for j in range(nbIter):
        z = nd.round(nd.random.uniform(0,1,(batch_size,k),ctx=myCtx))
        x = nd.dot(z,G)%2

        noiseBSC = nd.random.uniform(0.01,0.99,(batch_size,n),ctx=myCtx)
        noiseBSC = nd.floor(noiseBSC/nd.max(noiseBSC,axis=(1,)).reshape((batch_size,1)))

        y = (x + noiseBSC)%2

        with autograd.record():
            zHat = net(y,params)
            loss = SE(zHat,z)
        loss.backward()

        adam(params,vs,sqrs, lr, batch_size, t)
        t=t+1
        
        cumuLoss += loss.asscalar()
        zHat = nd.round(zHat)
        efficiency += nd.sum(nd.equal(zHat,z)).asscalar()

                
    Pc = efficiency/(batch_size*nbIter*k)
    Pe = 1 - Pc
    normCumuLoss = cumuLoss/(batch_size*nbIter*k)
    print("Epochs %d: Pe = %lf , loss = %lf" % (i,Pe,normCumuLoss))

