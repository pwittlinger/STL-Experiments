import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FixedLocator
import scipy as sp
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


def runTimeAnalysis(allDirs = os.listdir("./outputs/experiments"), formulas = ["Absence Test 1","Absence Test 2","Absence Test 3","Response Test 1","Response Test 2","Response Test 3"]):

    areFiles = [os.path.join(os.path.abspath(os.path.curdir),os.path.join("experiments",f)) for f in allDirs if (os.path.isfile(os.path.join(os.path.abspath(os.path.curdir),os.path.join("experiments",f))) == True)]
    fNames = [f.split("\\")[-1] for f in areFiles]
    head = ["runs", "avgTime", "std", "nTrials", "nObs", "timePerObs"]

    runData = []

    for f in areFiles:
        fname = f.split("\\")[-1]
        d = pd.read_csv(f, sep=";", header=None, names=head)
        r= d.groupby("nObs").agg({"runs":"sum"})

        for k in range(len(r)):
            r0 = r.iloc[k].str.replace(r"[", " ").str.replace(r"]"," ")
            r0Arr = r0.str.strip().str.split(" ").iloc[0]
            r0Runs = np.array([item for item in r0Arr if item]).astype(float)
            [runData.append([fname, r0.name, rItem]) for rItem in r0Runs]
        
    dfPlot = pd.DataFrame(runData, columns=["FileName", "nObs", "runTime"])
    dfPlot["logTime"] = np.log(dfPlot["runTime"])


    ##############################################################
    ########### Plot results of runtime experiments
    ##############################################################

    #plt.figure(figsize=(15,8))

    fig, axs_ = plt.subplots(1,2)
    fig.set_figheight(6)
    fig.set_figwidth(20)

    plotTitles = [
        "Absence Constraints",
        "Response Constraints"
    ]

    for i in range(2):
        axleg = []
        for j in range(int(len(formulas)/2)):
            axleg.append(f"{formulas[(j+i*3)]}")
            d = pd.read_csv(areFiles[(j+i*3)], sep=";", header=None, names=head)
            axs_[i].plot(d.loc[:,["nObs", "avgTime"]].groupby(by="nObs").mean().sort_values(by="nObs").index.values, d.loc[:,["nObs", "avgTime"]].groupby(by="nObs").mean().sort_values(by="nObs").values, marker=".", markersize=15, linewidth=3)
            axs_[i].set_ylabel("Time in Seconds")
            axs_[i].grid(True)
            axs_[i].set_xlabel("# Observations")
            #axs_[i].set_title(f"{i+1}) {plotTitles[i]}", y=-.15, pad=-25)
            axs_[i].set_title(f"{i+1}) {plotTitles[i]}")
            axs_[i].xaxis.set_minor_locator(FixedLocator([60, 60*60, 60*60*24, 60*60*24*2, 60*60*24*3, 60*60*24*5, 60*60*24*7, 60*60*24*10, 10**6]))
            axs_[i].ticklabel_format(style="plain")
            axs_[i].legend(axleg, fontsize="x-large")


    fig.tight_layout()
    #plt.show()
    plt.savefig("./outputs/performance.png", dpi=250)
    plt.savefig("./outputs/performance.pdf", dpi=250, format="pdf")

    ##############################################################
    ############ Calculate p-values to check if runtime is significantly different from two experiments
    ##############################################################
    
    for i in range(len(fNames)):
        for j in range(len(fNames)-i):
                if j==0: continue
                print(fNames[i], fNames[i+j], sp.stats.ks_2samp(np.array(dfPlot[dfPlot["FileName"] == fNames[i]]["runTime"]), np.array(dfPlot[dfPlot["FileName"] == fNames[i+j]]["runTime"]))[1])
                pval = sp.stats.ks_2samp(np.array(dfPlot[dfPlot["FileName"] == fNames[i]]["runTime"]), np.array(dfPlot[dfPlot["FileName"] == fNames[i+j]]["runTime"]))[1]
                with open("./outputs/resultsPvalues.txt", "a") as f:
                    f.write(f"{fNames[i]}; {fNames[i+j]}; {pval}\n")
    



if __name__=="__main__":
    formulas = [
    "Absence[0,m](x[t] > 1000)",
    "Absence[0,m](x[t] > 1000) ∧ Absence[0,m](x[t] < 150)",
    "Absence[0,m](x[t] > 1000) ∧ Absence[0,m](y[t] > 40)",
    "Response[0,m],[0,m](x[t]>1000, x[t]<400)",
    "Response[0,m],[0,1](|x’[t]| > 500, dis(bD)",
    "Response[0,m],[0,60](h[t] < 40, G[0,10](x[t])>300)"
]
    runTimeAnalysis(formulas=formulas)