import numpy as np
import matplotlib.pyplot as plt



def get_linear_curve(x, w, b=0,noise_scale=0):
    return w*x + b + noise_scale*np.random.randn(x.shape[0])

def forward(inputs, w, b):
    return w*inputs + b

def mse(Y, y):
    return (Y-y)**2

def back_propagation(inputs, outputs, targets, w, b, lr):
    dw = lr*(-2*inputs*(targets-outputs)).mean()
    db = lr*(-2*(targets-outputs)).mean()

    w -= dw
    b -= db
    return w, b

v = np.array([3, 4, 5])
u = np.array([3.2, 4.5, 4.8])

mse(v, u)

def model_fit(inputs, target, w, b, epochs = 200, lr = 0.001):
    for epoch in range(epochs):
        outputs = forward(inputs, w, b)
        loss = np.mean(mse(target, outputs))
        w, b = back_propagation(inputs, outputs, target, w, b, lr)

        if (epoch+1) % (epochs/10) == 0:
            print(f'Epoch: [{(epoch+1)}/{epochs}] Loss: [{loss:.4f}]')  
    return w, b

x = np.arange(-10, 10, 2)
Y = get_linear_curve(x, w = 1.8, b = 32)

w = np.random.rand(1)
b = np.zeros(1)

w, b = model_fit(x, Y, w, b, epochs=2000, lr=0.005)
print(f'W: {w[0]:.3f}, B: {b[0]:.3f}')

plt.style.use('dark_background')
plt.scatter(x, Y)
plt.plot(x, get_linear_curve(x, w, b), 'r', lw=2)
plt.show()
