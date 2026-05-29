import numpy as np
import matplotlib.pyplot as plt
# import plotext as graph

plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (10, 8)

def get_linear_curve(x, w, b=0,noise_scale=0):
    return w*x + b + noise_scale*np.random.randn(x.shape[0])

x = np.arange(-10, 30, 0.5)
Y = get_linear_curve(x, 1.8, 32, noise_scale=2.5)

print(x)
print(x.shape, Y)
print(x[-1])

#grafico em linha
plt.plot(x, Y)
# grafico pontolhado
plt.scatter(x, Y)
# nomeclatura dos vertices
plt.xlabel('°C', fontsize=10)
plt.ylabel('°F', fontsize=10)
plt.show()

# # grafico em barras verticais
# plt.hist(np.random.randn(300), bins=30, edgecolor='white')
# plt.show()

# graph.scatter(x[::5], Y[::5])
# graph.show()
# graph.savefig('graphic.png', dpi=150, bbox_inches='tight')
