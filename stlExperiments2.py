import sys
import os
os.environ["PATH"] += os.pathsep + './Graphviz/bin'
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.insert(0, 'src')


import stlcg
import stlviz as viz
#from stlcg import Expression
#from utils import print_learning_progress
#import graphviz
import torch
import numpy as np
import matplotlib.pyplot as plt
#import importlib
import pandas as pd
#import re
from re import fullmatch
from datetime import datetime
import pytz
#import timeit
from timeit import repeat
import gc
#os.environ["PATH"] += os.pathsep + './Graphviz/bin'
#sys.path.insert(0, 'src')

def runExperiment1(formula, data, nobs):
    print(f"Working on experiment {nobs}")
    co2Sens0 = np.array(data.iloc[0:(nobs),2]).reshape([1, nobs, 1])
    co2Tensor0 = torch.tensor(co2Sens0, requires_grad=False, dtype=torch.float16).flip(1)
    #humTensor0 = torch.tensor(humSens0, requires_grad=False, dtype=torch.float).flip(1)
    inputs = (co2Tensor0, co2Tensor0)
    pscale = 1     # "pscale" is the scale used for evaluting predicates
    scale = -1
    plt.plot(formula.robustness_trace(inputs, pscale=pscale, scale=scale).detach().numpy().reshape(1,nobs).squeeze())
    plt.xlabel("Number of Observation")
    plt.ylabel("Robustness value")
    plt.title(f"Satisfaction of STL formula over {nobs} observations")
    plt.savefig(os.path.join("./experiments/robustness",f"Response1_{nobs}.png"))
    plt.close()
    #r = timeit.repeat(lambda:formula.robustness_trace(inputs, pscale=pscale, scale=scale), number=n_experiments)
    r = repeat(lambda:formula.robustness_trace(inputs, pscale=pscale, scale=scale), number=n_experiments)
    ##r = timeit.repeat('formula.robustness_trace(inputs, pscale=pscale, scale=scale)', number=n_experiments)
    rScaled = np.divide(r,n_experiments)
    if os.path.isdir("./experiments"):
        #if os.path.isfile("./experiments/results.csv"):
        with open("./experiments/resultsResponse1.csv", "a") as f:
            f.write(f"{rScaled};{rScaled.mean()};{rScaled.std()};{n_experiments};{nobs};{rScaled.mean()/nobs}\n")
    return True

def runResponse2(formula, data, nobs):
    print(f"Discrete Response:\tWorking on experiment {nobs}")

    #co2Sens0 = np.array(data.iloc[0:(n),10]).reshape([1, n, 1])
    
    co2Sens0 = np.array(data.iloc[0:(nobs),data.columns.get_loc("co2Gradient")]).reshape([1, nobs, 1])
    #humSens0 = np.array(dataSensor0.iloc[0:(n),4]).reshape([1, n, 1])
    breakSens0 = np.array(data.iloc[0:(nobs),data.columns.get_loc("breakDown")]).reshape([1, nobs, 1])
    co2Tensor0 = torch.tensor(co2Sens0, requires_grad=False, dtype=torch.float).flip(1)
    breakTensor0 = torch.tensor(breakSens0, requires_grad=False, dtype=torch.float).flip(1)

    inputs = (co2Tensor0, breakTensor0)
    pscale = 1     # "pscale" is the scale used for evaluting predicates
    scale = -1
    plt.plot(formula.robustness_trace(inputs, pscale=pscale, scale=scale).detach().numpy().reshape(1,nobs).squeeze())
    plt.xlabel("Number of Observation")
    plt.ylabel("Robustness value")
    plt.title(f"Satisfaction of STL formula over {nobs} observations")
    plt.savefig(os.path.join("./experiments/robustness",f"Response2_{nobs}.png"))
    plt.close()
    r = repeat(lambda:formula.robustness_trace(inputs, pscale=pscale, scale=scale), number=n_experiments)
    rScaled = np.divide(r,n_experiments)
    if os.path.isdir("./experiments"):
        #if os.path.isfile("./experiments/results.csv"):
        with open("./experiments/resultsResponse2.csv", "a") as f:
            f.write(f"{rScaled};{rScaled.mean()};{rScaled.std()};{n_experiments};{nobs};{rScaled.mean()/nobs}\n")
    gc.collect()
    return True

def runResponseExperiment3(formula, data, nobs):
    print(f"Working on experiment {nobs}")
    co2Sens0 = np.array(data.iloc[0:(nobs),2]).reshape([1, nobs, 1])
    humSens0 = np.array(data.iloc[0:(nobs),4]).reshape([1, nobs, 1])
    co2Tensor0 = torch.tensor(co2Sens0, requires_grad=False, dtype=torch.float16).flip(1)
    humTensor0 = torch.tensor(humSens0, requires_grad=False, dtype=torch.float16).flip(1)
    inputs = (humTensor0, co2Tensor0)
    pscale = 1     # "pscale" is the scale used for evaluting predicates
    scale = -1
    plt.plot(formula.robustness_trace(inputs, pscale=pscale, scale=scale).detach().numpy().reshape(1,nobs).squeeze())
    plt.xlabel("Number of Observation")
    plt.ylabel("Robustness value")
    plt.title(f"Satisfaction of STL formula over {nobs} observations")
    plt.savefig(os.path.join("./experiments/robustness",f"Response3_{nobs}.png"))
    plt.close()
    r = repeat(lambda:formula.robustness_trace(inputs, pscale=pscale, scale=scale), number=n_experiments)
    rScaled = np.divide(r,n_experiments)
    if os.path.isdir("./experiments"):
        #if os.path.isfile("./experiments/results.csv"):
        with open("./experiments/resultsResponse3.csv", "a") as f:
            f.write(f"{rScaled};{rScaled.mean()};{rScaled.std()};{n_experiments};{nobs};{rScaled.mean()/nobs}\n")
    gc.collect()
    return True

def runAbsence1(formula, data, nobs):
    print(f"Absence1:\tWorking on experiment {nobs}")
    co2Sens0 = np.array(data.iloc[0:(nobs),2]).reshape([1, nobs, 1])
    co2Tensor0 = torch.tensor(co2Sens0, requires_grad=False, dtype=torch.float16).flip(1)
    inputs = (co2Tensor0)
    pscale = 1     # "pscale" is the scale used for evaluting predicates
    scale = -1
    plt.plot(formula.robustness_trace(inputs, pscale=pscale, scale=scale).detach().numpy().reshape(1,nobs).squeeze())
    plt.xlabel("Number of Observation")
    plt.ylabel("Robustness value")
    plt.title(f"Satisfaction of STL formula over {nobs} observations")
        #plt.savefig(os.path.join("./experiments/robustness", f"stlAlways1_{nobs}.png"))
    plt.savefig(os.path.join(expPlots, f"robustness/stlAlways1_{nobs}.png"))
    plt.close("all")

    r = repeat(lambda:formula.robustness_trace(inputs, pscale=pscale, scale=scale), number=n_experiments)
    rScaled = np.divide(r,n_experiments)
    if os.path.isdir("./experiments"):
        #if os.path.isfile("./experiments/results.csv"):
        with open("./experiments/resultsAbsence1.csv", "a") as f:
            f.write(f"{rScaled};{rScaled.mean()};{rScaled.std()};{n_experiments};{nobs};{rScaled.mean()/nobs}\n")

    gc.collect()

    return True

def runAbsence2(formula, data, nobs):
    print(f"Absence2:\tWorking on experiment {nobs}")
    co2Sens0 = np.array(data.iloc[0:(nobs),2]).reshape([1, nobs, 1])
        #humSens0 = np.array(dataSensor0.iloc[0:(nobs),4]).reshape([1, nobs, 1])
    co2Tensor0 = torch.tensor(co2Sens0, requires_grad=False, dtype=torch.float16).flip(1)
        #humTensor0 = torch.tensor(humSens0, requires_grad=False, dtype=torch.float).flip(1)
    inputs = (co2Tensor0,co2Tensor0)
    pscale = 1     # "pscale" is the scale used for evaluting predicates
    scale = -1
    plt.plot(formula.robustness_trace(inputs, pscale=pscale, scale=scale).detach().numpy().reshape(1,nobs).squeeze())
    plt.xlabel("Number of Observation")
    plt.ylabel("Robustness value")
    plt.title(f"Satisfaction of STL formula over {nobs} observations")
    plt.savefig(os.path.join("./experiments/robustness",f"stlAlways2_{nobs}.png"))
    plt.close("all")
    r = repeat(lambda:formula.robustness_trace(inputs, pscale=pscale, scale=scale), number=n_experiments)
    rScaled = np.divide(r,n_experiments)
    if os.path.isdir("./experiments"):
        #if os.path.isfile("./experiments/results.csv"):
        with open("./experiments/resultsAbsence2.csv", "a") as f:
            f.write(f"{rScaled};{rScaled.mean()};{rScaled.std()};{n_experiments};{nobs};{rScaled.mean()/nobs}\n")
    gc.collect()
    return True

def runAbsence3(formula, data, nobs):
    print(f"Working on experiment {nobs}")
    co2Sens0 = np.array(data.iloc[0:(nobs),2]).reshape([1, nobs, 1])
        #humSens0 = np.array(dataSensor0.iloc[0:(nobs),4]).reshape([1, nobs, 1])
    tempSens0 = np.array(data.iloc[0:(nobs),3]).reshape([1, nobs, 1])
    co2Tensor0 = torch.tensor(co2Sens0, requires_grad=False, dtype=torch.float16).flip(1)
    tempTensor0 = torch.tensor(tempSens0, requires_grad=False, dtype=torch.float16).flip(1)
        
        #humTensor0 = torch.tensor(humSens0, requires_grad=False, dtype=torch.float).flip(1)
    inputs = (co2Tensor0, tempTensor0)
    pscale = 1     # "pscale" is the scale used for evaluting predicates
    scale = -1

    plt.plot(formula.robustness_trace(inputs, pscale=pscale, scale=scale).detach().numpy().reshape(1,nobs).squeeze())
    plt.xlabel("Number of Observation")
    plt.ylabel("Robustness value")
    plt.title(f"Satisfaction of STL formula over {nobs} observations")
    plt.savefig(os.path.join("./experiments/robustness",f"stlAlways3_{nobs}.png"))
    plt.close("all")
        #r = timeit.repeat(lambda:formula.robustness_trace(inputs, pscale=pscale, scale=scale), number=n_experiments)
    r = repeat(lambda: formula.robustness_trace(inputs, pscale=pscale, scale=scale), number=n_experiments)
    rScaled = np.divide(r,n_experiments)
    if os.path.isdir("./experiments"):
    #if os.path.isfile("./experiments/results.csv"):
        with open("./experiments/resultsAbsence3.csv", "a") as f:
            f.write(f"{rScaled};{rScaled.mean()};{rScaled.std()};{n_experiments};{nobs};{rScaled.mean()/nobs}\n")   
    gc.collect()
    return True

def isWeekend(dateInDTformat):
    return(dateInDTformat.weekday()>4)

if __name__ == "__main__":
    #print(True)
    print(dname)

    data = pd.read_csv(r"./data/co2_dump151018.csv", header=None)
    data = data[np.logical_not(np.logical_not(data[0].str.isnumeric()))]
    bm = data.iloc[:,2].astype(str).str.contains(r'^[\d]+.[\d]+$', regex=True)
    data2 = data[bm]
    data2.iloc[:,2] = data2[2].apply(lambda f: float(str(f).replace("..", ".")))
    data2.iloc[:,3] = data2[3].apply(lambda f: float(str(f).replace("e+","")))
    data2.iloc[:,4] = data2[4].apply(lambda f: float(str(f).replace("e+","")))
    data2 = data2[np.logical_not(data2[6].isna())]
    bm5 = (data2.iloc[:,5].apply(lambda f: (fullmatch(r'^([\d]*.?[\d]*)$', str(f))))).astype(bool)
    data3 = data2[bm5]
    data = data3.astype("float32")
    data[1] = data3.iloc[:,1].astype('Int64')
    #########
    iqrco2 = (data[2].quantile(.75)-data[2].quantile(.25))
    iqrtemp = (data[3].quantile(.75)-data[3].quantile(.25))
    data.sort_values(by=0, inplace=True)
    data.drop(index=[0,1,2], inplace=True)
    ################
    data = data[(data[3]< (data[3].quantile(.75)+1.5*iqrtemp)) & (data[3]> (data[3].quantile(.25)-1.5*iqrtemp))]
    data = data[(data[4]>0) & ((data[4]<=100))]
    data = data[data[2]<2500]
    data[7] = data[1].apply(lambda f: datetime.fromtimestamp(f, tz=pytz.timezone("Asia/Jakarta")).strftime("%d-%m-%Y %H:%M:%S"))
    data.iloc[:,7] = data.iloc[:,7].apply(lambda f: datetime.strptime(f, "%d-%m-%Y %H:%M:%S"))
    data.sort_values(1, inplace=True)
    print("preprocessing done")
    n = 60*60
    fig, axs_ = plt.subplots(3)
    fig.set_figheight(12)
    fig.set_figwidth(15)

    for k in [0,1,9]:
        axs_[[0,1,9].index(k)].plot(range(n),data[data[6]==k].iloc[0:n,2], ".-", linewidth=3, markersize=15, label="x")
        #plt.plot(range(n), np.full(n,300))
        axs_[[0,1,9].index(k)].plot(range(n), np.full(n,300))
        axs_[[0,1,9].index(k)].set_title(f"First {n} observations for Sensor {k}")
        

    #plt.title("CO2 Concentration")
    fig.suptitle("CO2 Concentration in ppm")
    #plt.show()
    fpPlots = os.path.join(dname,"./plots")
    if np.logical_not(os.path.isdir(fpPlots)):
        os.mkdir(os.path.join(dname,"/plots"))
    
    plt.savefig(os.path.join(fpPlots,"co2ppm.png"))
    #plt.show()

    fig, axs_ = plt.subplots(3)
    fig.set_figheight(12)
    fig.set_figwidth(15)

    for k in [0,1,9]:
        axs_[[0,1,9].index(k)].plot(range(n),data[data[6]==k].iloc[0:n,3], ".-", linewidth=3, markersize=15, label="x")
        axs_[[0,1,9].index(k)].set_title(f"First {n} observations for Sensor {k}")
        #plt.plot(range(n), np.full(n,300))
        #axs_[[0,1,9].index(k)].plot(range(n), np.full(n,300))

    #plt.title("CO2 Concentration")
    fig.suptitle("Temperature by Sensor")
    #plt.show()
    #plt.savefig(os.path.join("./plots","temp.png"))
    plt.savefig(os.path.join(fpPlots,"temp.png"))

    fig, axs_ = plt.subplots(3)
    fig.set_figheight(12)
    fig.set_figwidth(15)

    for k in [0,1,9]:
        axs_[[0,1,9].index(k)].plot(range(n),data[data[6]==k].iloc[0:n,4], ".-", linewidth=3, markersize=15, label="x")
        axs_[[0,1,9].index(k)].set_title(f"First {n} observations for Sensor {k}")
        #plt.plot(range(n), np.full(n,300))
        #axs_[[0,1,9].index(k)].plot(range(n), np.full(n,300))

    #plt.title("CO2 Concentration")
    fig.suptitle("Humidity in % by Sensor")
    #plt.show()
    #plt.savefig(os.path.join("./plots","humidity.png"))
    plt.savefig(os.path.join(fpPlots,"humidity.png"))

    fig, axs_ = plt.subplots(3)
    fig.set_figheight(20)
    fig.set_figwidth(15)

    for k in [0,1,9]:
        axs_[[0,1,9].index(k)].plot(range(n),np.diff(data[data[6]==k].iloc[0:n+1,1]), "-", linewidth=3, markersize=15, label="x")
        axs_[[0,1,9].index(k)].set_title(f"First {n} observations for Sensor {k}")

    #plt.title("CO2 Concentration")
    fig.suptitle("Time lag in Seconds by Sensor")
    #plt.show()
    #plt.savefig(os.path.join("./plots","timelag.png"))
    plt.savefig(os.path.join(fpPlots,"timelag.png"))

    print("Plotting done")

    dataSensor0 = data[data[6] == 0]
    dataSensor1 = data[data[6] == 1]
    dataSensor9 = data[data[6] == 9]

    co2Gradient0 = np.concatenate([np.diff(dataSensor0.sort_values(by=0)[2]),[0]], axis=0)
    #dataSensor0.iloc[:,10] = (pd.Series(np.abs(co2Gradient0)))
    dataSensor0["co2Gradient"] = (pd.Series(np.abs(co2Gradient0)))
    #dataSensor0.loc[:,"co2Gradient"] = (pd.Series(np.abs(co2Gradient0)))
    dataSensor0.iloc[:, dataSensor0.columns.get_loc("co2Gradient")] = pd.Series(np.abs(co2Gradient0))
    timelag0 = np.concatenate([[0],np.diff(dataSensor0[1].sort_values())], axis=0)
    #dataSensor0.loc[:,"timeLag"] = timelag0
    dataSensor0.loc[:,"timeLag"] = timelag0
    #dataSensor0.insert(10, "co2Grad", pd.Series(np.abs(co2Gradient0)))

    weekendBM = dataSensor0.iloc[:,7].apply(lambda f: isWeekend(f))
    breakdown = (weekendBM) & (dataSensor0["timeLag"]>1000)
    dataSensor0["breakDown"] = breakdown.astype(int)
    dataSensor0.iloc[:, dataSensor0.columns.get_loc("breakDown")] = breakdown.astype(int)



    ##################################


    #n_obs = [60, 60*60, 60*60*24, 60*60*24*2, 60*60*24*3, 60*60*24*5, 60*60*24*7, 60*60*24*10]
    #n_obs = [60, 60*60, 60*60*24, 60*60*24*2]
    n_obs = [60, 60*60, 60*60*24, 60*60*24*2, 60*60*24*3, 60*60*24*5, 60*60*24*7, 60*60*24*10, 10**6]
    n_experiments = 1

    
    ######################################
    c = torch.tensor(1000.0, dtype=torch.float16, requires_grad=False)
    d = torch.tensor(400.0, dtype=torch.float16, requires_grad=False)


    ϕCo2High = stlcg.GreaterThan(lhs='x', val=c)
    ϕCo2Med = stlcg.LessThan(lhs='x', val=d)
    #ϕ3 = stlcg.LessThan(lhs='w', val=d)
    #ϕ = stlcg.Always(subformula=ϕ1)
    #ψ = stlcg.Eventually(subformula=ϕ3)
    formula = stlcg.Implies(subformula1=ϕCo2High, subformula2=stlcg.Eventually(ϕCo2Med))
    #formula = stlcg.Implies(subformula1=ϕHum, subformula2=ϕCo2)
    #ftest = stlcg.Always(subformusla=formula, interval=[0,60])
    print(formula)
    #viz.make_stl_graph(formula).render("./plots/formulae/stlFormulaResponse", format="png")
    graph = viz.make_stl_graph(formula)
    viz.save_graph(graph, "./plots/formulae/stlFormulaResponse1")
    plt.close("all")
    

    #[runExperiment1(formula, dataSensor0, nobs) for nobs in n_obs]

    ############################
    c = torch.tensor(.99, dtype=torch.float, requires_grad=False)
    d = torch.tensor(500.0, dtype=torch.float, requires_grad=False)

    ϕBreak = stlcg.GreaterThan(lhs='x', val=c)
    ϕCo2 = stlcg.GreaterThan(lhs='g', val=d)

    formula = stlcg.Implies(subformula1=ϕCo2, subformula2=stlcg.Eventually(ϕBreak, interval=[0,1]))
    print(formula)
    graph = viz.make_stl_graph(formula)
    viz.save_graph(graph, "./plots/formulae/stlFormulaResponseDiscrete")
    plt.close("all")

    [runResponse2(formula, dataSensor0, nobs) for nobs in n_obs]
    
    ###########################
    c = torch.tensor(40.0, dtype=torch.float16, requires_grad=False)
    d = torch.tensor(300.0, dtype=torch.float16, requires_grad=False)

    ϕHum = stlcg.LessThan(lhs='x', val=c)
    ϕCo2 = stlcg.LessThan(lhs='w', val=d)
    #ϕ3 = stlcg.LessThan(lhs='w', val=d)
    #ϕ = stlcg.Always(subformula=ϕ1)
    #ψ = stlcg.Eventually(subformula=ϕ3)
    formula = stlcg.Implies(subformula1=ϕHum, subformula2=stlcg.Eventually(stlcg.Always(ϕCo2, interval=[0,10]), interval=[0,60]))
    #formula = stlcg.Implies(subformula1=ϕHum, subformula2=ϕCo2)
    #ftest = stlcg.Always(subformusla=formula, interval=[0,60])
    print(formula)
    #viz.make_stl_graph(formula).render("./plots/formulae/stlFormulaResponse", format="png")
    graph = viz.make_stl_graph(formula)
    viz.save_graph(graph, "./plots/formulae/stlFormulaResponse3")
    plt.close("all")
    
    #[runResponseExperiment3(formula, dataSensor0, nobs) for nobs in n_obs]
    print("First experiment done")
    ############################################
    gc.collect()
    u = torch.tensor(1000.0, dtype=torch.float16, requires_grad=False)
    ϕCo2 = stlcg.GreaterThan(lhs='x', val=u)

    #os.mkdir("./experiments")
    #os.mkdir("./experiments/robustness")

    formula = stlcg.Negation(stlcg.Eventually(subformula=ϕCo2))
    viz.save_graph(viz.make_stl_graph(formula), "./plots/formulae/stlFormulaAlways")

    expPlots = os.path.join(dname,"./experiments")
    if np.logical_not(os.path.isdir(expPlots)):
        os.mkdir(os.path.join(dname,"./experiments"))
    if np.logical_not(os.path.isdir(os.path.join(expPlots, "robustness"))):
        os.mkdir(os.path.join(expPlots, "robustness"))

    #[runAbsence1(formula, dataSensor0, nobs) for nobs in n_obs]
    ############################################
    
    u = torch.tensor(1000.0, dtype=torch.float, requires_grad=False)
    lower =  torch.tensor(150.0, dtype=torch.float, requires_grad=False)

    #ϕHum = stlcg.LessThan(lhs='x', val=c)
    #ϕCo2 = stlcg.LessThan(lhs='x', val=u)
    #ϕCo2low = stlcg.GreaterThan(lhs='x', val=lower)

    ϕCo2 = stlcg.Negation(stlcg.Eventually(stlcg.GreaterThan(lhs='x', val=u)))
    ϕCo2low = stlcg.Negation(stlcg.Eventually(stlcg.LessThan(lhs='x', val=lower)))

    #formula = stlcg.Implies(subformula1=ϕHum, subformula2=stlcg.Eventually(stlcg.Always(ϕCo2, interval=[0,10]), interval=[0,60]))
    #formula = stlcg.Always(subformula=stlcg.And(ϕCo2, ϕCo2low))
    formula = stlcg.And(ϕCo2, ϕCo2low)
    #ftest = stlcg.Always(subformula=formula, interval=[0,60])
    print(formula)
    viz.save_graph(viz.make_stl_graph(formula), "./plots/formulae/stlFormulaAlways2")

    #[runAbsence2(formula, dataSensor0, nobs) for nobs in n_obs]


    ##################################
    u = torch.tensor(1000.0, dtype=torch.float16, requires_grad=False)
    lower =  torch.tensor(150.0, dtype=torch.float16, requires_grad=False)
    tUp = torch.tensor(40.0, dtype=torch.float16, requires_grad=False)

    #ϕHum = stlcg.LessThan(lhs='x', val=c)
    #ϕCo2 = stlcg.LessThan(lhs='x', val=u)
    #ϕCo2low = stlcg.GreaterThan(lhs='x', val=lower)
    #ϕtempUp = stlcg.Always(stlcg.LessThan(lhs='t', val=tUp))

    ϕCo2 = stlcg.Negation(stlcg.Eventually(stlcg.GreaterThan(lhs='x', val=u)))
    #ϕCo2low = stlcg.GreaterThan(lhs='x', val=lower)
    ϕtempUp = stlcg.Negation(stlcg.Eventually(stlcg.GreaterThan(lhs='t', val=tUp)))


    #f1 = stlcg.Always(subformula=stlcg.And(ϕCo2, ϕCo2low))
    formula = stlcg.And(subformula1 = ϕCo2, subformula2 = ϕtempUp)
    print(formula)
    viz.save_graph(viz.make_stl_graph(formula), "./plots/formulae/stlFormulaAlways3")

    #[runAbsence3(formula, dataSensor0, nobs) for nobs in n_obs]
    ####

    print(True)