import numpy as np

if __name__ == "__main__":
    x = np.arange(1,10,0.001)
    y = 1 + (x * 2) + (np.random.normal(0, 1, len(x)) * 5)

    mx = np.mean(x)
    my = np.mean(y)

    c1 = np.sum((x - mx) * (y - my))/np.sum((x - mx)**2)
    c0 = my - c1 * mx

    print c0, c1