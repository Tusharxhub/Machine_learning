import numpy as np
from matplotlib import pyplot as plt

np.random.seed(42)
x = np.array(np.random.uniform(0, 10, 50))
y = np.array(np.random.uniform(0, 10, 50))


def grad(x, y):
    m_corr = c_corr = 0
    lr = 0.01
    ite = 100
    n = len(x)
    
    for i in range(ite):
        y_pred = m_corr * x + c_corr
        cost = np.sum((y_pred - y) ** 2) / n
        dm = (2 / n) * np.sum(x * (y_pred - y))
        db = (2 / n) * np.sum(y_pred - y)
        
        m_corr = m_corr - lr * dm
        c_corr = c_corr - lr * db
        print(f"m_corr: {m_corr}, c_corr: {c_corr}, cost: {cost}")
    
    return m_corr, c_corr

m, c = grad(x, y)


plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Actual Data', color='blue', alpha=0.6)
y_final = m * x + c
plt.plot(x, y_final, label=f'Linear Fit (y={m:.2f}x+{c:.2f})', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression using Gradient Descent')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/home/tushardevx01/Documents/Machine learning/Exam/graph.png', dpi=300, bbox_inches='tight')
print("Graph saved as graph.png")
plt.show()