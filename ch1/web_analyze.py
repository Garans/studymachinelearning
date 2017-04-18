import scipy as sp
import matplotlib.pyplot as plt

colors = ['g', 'k', 'b', 'm', 'r']
line_styles = ['-', '-.', '--', ':', '-']


def error(f, data_x, data_y):
    return sp.sum((f(data_x) - data_y) ** 2)


def window(data_x, data_y, filename, models=None, mx=None):
    plt.figure(num=None, figsize=(8, 6))
    plt.clf()
    plt.scatter(data_x, data_y, s=10)
    plt.title("Web traffic over the last month")
    plt.xlabel("Time")
    plt.ylabel("Hits/hours")
    plt.xticks([w * 7 * 24 for w in range(10)], ['week {}'.format(w) for w in range(10)])

    if models:
        if mx is None:
            mx = sp.linspace(0, x[-1], 1000)

        for model, style, color in zip(models, line_styles, colors):
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)

        plt.legend(["d={}".format(m.order) for m in models], loc="upper left")

    plt.autoscale(tight=True)
    plt.grid(True, linestyle='-', color='0.75')
    plt.savefig("charts/{}.png".format(filename))

data = sp.genfromtxt('data/web_traffic.tsv', delimiter="\t")

x = data[:, 0]
y = data[:, 1]

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

window(x, y, '0')

fp_1, residuals, rank, sv, r_cond = sp.polyfit(x, y, 1, full=True)

print("Model parameters: {}".format(fp_1))
print("Res: {}".format(residuals))

f1 = sp.poly1d(fp_1)
print("Error: {}".format(error(f1, x, y)))
print("\n-------")

window(x, y, '1', [f1])

fp_2, residuals_2, rank_2, sv_2, r_cond_2 = sp.polyfit(x, y, 2, full=True)

print("Model parameters: {}".format(fp_2))
print("Res: {}".format(residuals_2))

f2 = sp.poly1d(fp_2)

window(x, y, '2', [f1, f2])
window(x, y, '3', [f1, f2] + [sp.poly1d(sp.polyfit(x, y, i)) for i in (3, 10, 100)])

inflection = int(3.5 * 7 * 24)

xa = x[:inflection].astype(int)
ya = y[:inflection].astype(int)

xb = x[inflection:].astype(int)
yb = y[inflection:].astype(int)

fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))


# e = error(fa, xa, ya)
# print("Error inflection={}".format((fa + e)))
#
# e = error(fb, xb, yb)
# print("Error inflection={}".format((fb + e)))

window(x, y, '4', [fa, fb])

