import matplotlib
matplotlib.rcParams.update({'font.size': 12})
from matplotlib import pyplot as plt
import numpy as np
from collections import OrderedDict
from scipy import stats

from plot_assemble_results import *

results_folder = "../calculated_results/normalized_neuronwise_lenet_cifar10_loss01_comp2"

prefix_len = 5
gap_normalizer = 50000.0

tracial_measures, meig_measures, trace, fisher_rao, pacbayes_flat, pacbayes_orig, norm, gaps, labels = assemble_compare_measures(results_folder, "reparametrized_comparison_measures", 
                    prefix_len = prefix_len, gap_normalizer = gap_normalizer, filter_out= 'wd0.0001')

colors = plt.cm.viridis(np.linspace(0, 0.8, 6))


fig, host = plt.subplots()
fig.set_size_inches(5*1.6, 5)
par1 = host.twiny()
sc1 = host.scatter(tracial_measures, gaps, label='Relative flatness', color=colors[0], marker='o', alpha=0.7)
sc2 = host.scatter(trace, gaps, label='Trace', color=colors[1], marker='p', alpha=0.7)
#sc3 = par1.scatter(fisher_rao, gaps, label='Fisher-Rao norm', color=colors[2], marker='X', alpha=0.7) ## without reparametrization
sc3 = host.scatter(fisher_rao, gaps, label='Fisher Rao norm', color=colors[2], marker='X', alpha=0.7) ## with reparametrization
sc4 = par1.scatter(norm, gaps, label='Weight norm', color=colors[3], marker='d', alpha=0.7)
#sc5 = par1.scatter(pacbayes_flat, gaps, label='PacBayes flatness', color=colors[4], marker='*', alpha=0.7)
#sc6 = host.scatter(pacbayes_orig, gaps, label='PacBayes', color=colors[5], marker='s', alpha=0.7) ## without reparametrization
sc6 = par1.scatter(pacbayes_orig, gaps, label='PacBayes', color=colors[5], marker='s', alpha=0.7) ## with reparametrization
    

scs = [sc1, sc2, sc3, sc4, sc6]

slope, intercept, rval, _, _ = stats.linregress(tracial_measures, gaps)
#host.annotate(r"$\rho$ = %.2f"%rval, xy=(9000, 6.7), c=colors[0]) ## without reparametrizations
host.annotate(r"$\rho$ = %.2f"%rval, xy=(8000, 7.5), c=colors[0]) ## with reparametrizations
print(rval)
#x_line = np.linspace(500,12000,100) ## without reparametrizations
x_line = np.linspace(1e3,1.2e4,100) ## with reparametrizations
y_line = slope * x_line + intercept
host.plot(x_line, y_line, ls="--", c=colors[0])

slope, intercept, rval, _, _ = stats.linregress(norm, gaps)
#par1.annotate(r"$\rho$ = %.2f"%rval, xy=(60, 4), c=colors[3]) ## without reparametrizations
par1.annotate(r"$\rho$ = %.2f"%rval, xy=(1e6, 7.7), c=colors[3]) ## with reparametrizations
print(rval)
#x_line = np.linspace(65,90,100) ## without reparametrizations
x_line = np.linspace(-1e5,0.9e6,100) ## with reparametrizations
y_line = slope * x_line + intercept
par1.plot(x_line, y_line, ls="--", c=colors[3])

slope_p, intercept_p, rval_p, _, _ = stats.linregress(trace, gaps)
#host.annotate(r"$\rho$ = %.2f"%rval_p, xy=(2000, 7.5), color=colors[1]) ## without reparametrizations
host.annotate(r"$\rho$ = %.2f"%rval_p, xy=(20, 7.3), color=colors[1]) ## with reparametrizations
print(rval_p)
#x_line_p = np.linspace(-100,2000,100) ## without reparametrizations
x_line_p = np.linspace(0,50,100) ## with reparametrizations
y_line_p = slope_p * x_line_p + intercept_p
host.plot(x_line_p, y_line_p, ls="--", color=colors[1])

slope_p, intercept_p, rval_p, _, _ = stats.linregress(fisher_rao, gaps)
#par1.annotate(r"$\rho$ = %.2f"%rval_p, xy=(52, 6.5), color=colors[2]) ## without reparametrizations
host.annotate(r"$\rho$ = %.2f"%rval_p, xy=(90, 6.7), color=colors[2]) ## with reparametrizations
print(rval_p)
#x_line_p = np.linspace(10,60,100) ## without reparametrizations
x_line_p = np.linspace(10,70,100) ## with reparametrizations
y_line_p = slope_p * x_line_p + intercept_p
#par1.plot(x_line_p, y_line_p, ls="--", color=colors[2]) ## without reparametrizations
host.plot(x_line_p, y_line_p, ls="--", color=colors[2]) ## with reparametrizations

#slope_p, intercept_p, rval_p, _, _ = stats.linregress(pacbayes_flat, gaps)
#par1.annotate(r"$\rho$ = %.2f"%rval_p, xy=(0.5, 4.5), color=colors[4]) ## without reparametrizations
#print(rval_p)
#x_line_p = np.linspace(0.00,0.027,100) ## without reparametrizations
#y_line_p = slope_p * x_line_p + intercept_p
#par1.plot(x_line_p, y_line_p, ls="--", color=colors[4])

slope_p, intercept_p, rval_p, _, _ = stats.linregress(pacbayes_orig, gaps)
#host.annotate(r"$\rho$ = %.2f"%rval_p, xy=(18000, 7.0), color=colors[5]) ## without reparametrizations
par1.annotate(r"$\rho$ = %.2f"%rval_p, xy=(1.25e7, 5.7), color=colors[5]) ## with reparametrizations
print(rval_p)
#x_line_p = np.linspace(5000,23000,100) ## without reparametrizations
x_line_p = np.linspace(0,1.75e7,100) ## with reparametrizations
y_line_p = slope_p * x_line_p + intercept_p
#host.plot(x_line_p, y_line_p, ls="--", color=colors[5]) ## without reparametrizations
par1.plot(x_line_p, y_line_p, ls="--", color=colors[5]) ## with reparametrizations


#host.set_xlabel("Realtive flatness & Trace & PacBayes") ## without reparametrization
host.set_xlabel("Realtive flatness & Trace & Fisher-Rao norm") ## with reparametrization
#par1.set_xlabel("Fisher-Rao norm & Weight norm") ## without reparametrization
par1.set_xlabel("PacBayes & Weight norm") ## with reparametrization
host.set_ylabel("Generalization gap")
host.set_xscale("log") ## with reparametrization
#par1.legend(scs, [sc.get_label() for sc in scs], loc='lower center') ## without reparametrization
par1.legend(scs, [sc.get_label() for sc in scs], loc='lower right') ## with reparametrization
plt.show()
