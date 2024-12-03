import sys
import os
os.environ["PATH"] += os.pathsep + './Graphviz/bin'
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.insert(0, 'src')

import stlcg
import stlviz as viz
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from re import fullmatch
from datetime import datetime
import pytz
from timeit import repeat
import gc
from declare2stl_parser import Parser

if np.logical_not(os.path.isdir(os.path.join(dname,"outputs"))):
    os.mkdir(os.path.join(dname,"outputs"))
    os.mkdir(os.path.join(dname,"outputs/experiments"))
    os.mkdir(os.path.join(dname,"outputs/plots"))
    os.mkdir(os.path.join(dname,"outputs/formulas"))

def plotData(data, colName, figTitle, nobs=3600, dname=dname):

    colmap = {"co2":2,
              "temp":3,
              "humidity":4,
              "light":5
     }
    
    fig, axs_ = plt.subplots(3)
    fig.set_figheight(12)
    fig.set_figwidth(15)

    for k in [0,1,9]:
        axs_[[0,1,9].index(k)].plot(range(nobs),data[data[6]==k].iloc[0:nobs,colmap[colName]], ".-", linewidth=3, markersize=15, label="x")
        axs_[[0,1,9].index(k)].set_title(f"First {nobs} observations for Sensor {k}")
        
    fig.suptitle(figTitle)
    fpPlots = os.path.join(dname,"./outputs/plots")
    if np.logical_not(os.path.isdir(fpPlots)):
        os.mkdir(os.path.join(dname,"/outputs/plots"))
    plt.savefig(os.path.join(fpPlots,f"{colName}.png"))
    plt.close()
    return True

def isWeekend(dateInDTformat):
    return(dateInDTformat.weekday()>4)

def dataPreprocessing(pathToData=r"./data/co2_dump151018.csv"):

    data = pd.read_csv(pathToData, header=None)
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
    data[1] = data3.iloc[:,1].astype('Int64') # Makes handling of unix timestamps easier

    #########
    # Removing massive outliers in measurements
    #iqrco2 = (data[2].quantile(.75)-data[2].quantile(.25))
    iqrtemp = (data[3].quantile(.75)-data[3].quantile(.25))
    data.sort_values(by=0, inplace=True)
    data.drop(index=[0,1,2], inplace=True)
    ################
    data = data[(data[3]< (data[3].quantile(.75)+1.5*iqrtemp)) & (data[3]> (data[3].quantile(.25)-1.5*iqrtemp))]
    data = data[(data[4]>0) & ((data[4]<=100))]
    data = data[data[2]<2500]
    # Converting the unix timestamp to datetime -> checking if it's a weekday later
    data[7] = data[1].apply(lambda f: datetime.fromtimestamp(f, tz=pytz.timezone("Asia/Jakarta")).strftime("%d-%m-%Y %H:%M:%S"))
    data.iloc[:,7] = data.iloc[:,7].apply(lambda f: datetime.strptime(f, "%d-%m-%Y %H:%M:%S"))
    data.sort_values(1, inplace=True)
    return data

def createInputTensors(data, nobs, inSignInd):
    """ Return a tensor for the specified column with the correct length
    """
    if isinstance(inSignInd, str) & (inSignInd in colmap):
        indSignalColumn = colmap[inSignInd]
    elif isinstance(inSignInd, str) & np.logical_not(inSignInd in colmap.keys()):
        indSignalColumn = data.columns.get_loc(inSignInd)
    elif isinstance(inSignInd, int):
        indSignalColumn = inSignInd
        
    tens_ = np.array(data.iloc[0:(nobs),indSignalColumn]).reshape([1, nobs, 1])

    return torch.tensor(tens_, requires_grad=False, dtype=torch.float16).flip(1)

def runExperiment(formula, data, nobs, inSign, experimentName, pscale=1, scale=-1, n_experiments=1):
    print(formula)
    graph = viz.make_stl_graph(formula)
    viz.save_graph(graph, f"./outputs/formulas/{experimentName}")


    print(f"{experimentName}:\tWorking on experiment {nobs}")

    if len(inSign) == 1:
        inputs=(createInputTensors(data, nobs, inSign[0]))
    else: inputs=(createInputTensors(data, nobs, inSign[0]),)
    
    for i in range(len(inSign)-1):
        inputs+=(createInputTensors(data, nobs, inSign[i+1]),)

    plt.plot(formula.robustness_trace(inputs, pscale=pscale, scale=scale).detach().numpy().reshape(1,nobs).squeeze())
    plt.xlabel("# Observation")
    plt.ylabel("Robustness value")
    plt.title(f"Satisfaction of STL formula over {nobs} observations")
    
    pname = os.path.join(os.path.join(os.path.join(dname, "outputs"), "experiments"), "robustness")
    if not os.path.exists(pname):
        os.makedirs(pname)
    plt.savefig(os.path.join(pname,f"{experimentName}_{nobs}.png"))
    plt.close()

    r = repeat(lambda:formula.robustness_trace(inputs, pscale=pscale, scale=scale), number=n_experiments)
    rScaled = np.divide(r,n_experiments) # only relevant in case more than 1 experiment is run at the time, as timeit returns the absolute runtime, not per iteration

    if os.path.isdir("./outputs/experiments"):
        fullFilePathResults = os.path.join(os.path.join(os.path.join(dname, "outputs"), "experiments"), f"results{experimentName}.csv")
        #with open(f"./outputs/experiments/results{experimentName}.csv", "a") as f:
        with open(fullFilePathResults, "a") as f:
            f.write(f"{rScaled};{rScaled.mean()};{rScaled.std()};{n_experiments};{nobs};{rScaled.mean()/nobs}\n")

    return rScaled.mean()

if __name__ == "__main__":

    #print(dname)

    data = dataPreprocessing()
    # loading the data and basic preprocessing
    # only keeping data that contains floats for measurements + discarding empty rows
    print("preprocessing done")
    #########################################

    plotColumns = ["co2", "temp", "humidity", "light"]
    pltTitles = [
        "CO2 Emissions in ppm",
        "Humidity in %",
        "Temperature in Degree Celsius",
        "Light Intensity per Sensor"
    ]

    [plotData(data, clN, pltTitles[iCn]) for iCn,clN in enumerate(plotColumns)]


    ############################################################################
    ######### Splitting the data per Sensor
    ######### All experiments run on Sensor
    ############################################################################

    dataSensor0 = data[data[6] == 0]
    #dataSensor1 = data[data[6] == 1]
    #dataSensor9 = data[data[6] == 9]

    co2Gradient0 = np.concatenate([np.diff(dataSensor0.sort_values(by=0)[2]),[0]], axis=0)
    dataSensor0["co2Gradient"] = (pd.Series(np.abs(co2Gradient0)))
    dataSensor0.iloc[:, dataSensor0.columns.get_loc("co2Gradient")] = pd.Series(np.abs(co2Gradient0))
    timelag0 = np.concatenate([[0],np.diff(dataSensor0[1].sort_values())], axis=0)
    dataSensor0.loc[:,"timeLag"] = timelag0

    weekendBM = dataSensor0.iloc[:,7].apply(lambda f: isWeekend(f))
    breakdown = (weekendBM) & (dataSensor0["timeLag"]>1000)
    dataSensor0["breakDown"] = breakdown.astype(int)
    dataSensor0.iloc[:, dataSensor0.columns.get_loc("breakDown")] = breakdown.astype(int)



    ####################################################################################################################################################################################################################################
    ########################### Starting with the Experiments
    ########################### Setting hyper parameters

    n_obs = [60, 60*60, 60*60*24, 60*60*24*2, 60*60*24*3, 60*60*24*5, 60*60*24*7, 60*60*24*10, 10**6]
    n_experiments = 1

    intervalLengths = [1,2,5,10,50]
    inSignals = [["co2"], ["co2", "co2"], ["co2", "temp"], ["co2", "co2"], ["co2Gradient", "breakDown"]]
    expNames = ["Absence1Loop", "Absence2Loop", "Absence3Loop", "Response1Loop", "Response2Loop"]
    colmap = {"co2":2,
                "temp":3,
                "humidity":4,
                "light":5}
    
    if np.logical_not(os.path.isdir(os.path.join(dname,"outputs"))):
        outPath = os.path.join(dname,"outputs")
        os.makedirs(outPath)
        os.makedirs(os.path.join(outPath,"experiments"))
        os.makedirs(os.path.join(outPath,"plots"))
        os.makedirs(os.path.join(outPath,"formulas"))

    
    ############################################################################
    ######### Response Test 1 : Unbounded Interval on same Input Signal
    ############################################################################
    with open("./Constraints.txt", "r", encoding='utf-8') as f:
        constraintStrings = [line.rstrip() for line in f]
    singleExperiments = [cS_.split(" ∧ ") for cS_ in constraintStrings[:-1]]

    stlPars = Parser()

    for itv_ in intervalLengths:

        exprmnts= []
        for e in singleExperiments:
            print(e)
            forms = [stlPars.transform_declare2STLCG(se.replace(",m]", f",{itv_}]")) for se in e]
            if len(forms)>1:
                exprmnts.append(stlcg.And(forms[0], forms[1]))
            else:
                exprmnts.append(forms[0])

        for o,p,q in zip(exprmnts, inSignals, expNames):
            print(o,p,q)
            
            for nobs in n_obs:
                runExperiment(formula = o, data=dataSensor0, nobs=nobs, inSign=p, experimentName=q+f"_{itv_}")

    ############################################################################
    ######### Response Test 1 : Unbounded Interval on same Input Signal
    ############################################################################                

    # The last / nested Response formula is not handled by the Parser. That's why I'm creating that manually now.

    c = torch.tensor(40.0, dtype=torch.float16, requires_grad=False)
    d = torch.tensor(300.0, dtype=torch.float16, requires_grad=False)

    ϕHum = stlcg.LessThan(lhs='x', val=c)
    ϕCo2 = stlcg.LessThan(lhs='w', val=d)


    formula = stlcg.Implies(subformula1=ϕHum, subformula2=stlcg.Eventually(stlcg.Always(ϕCo2, interval=[0,10]), interval=[0,60]))

    [runExperiment(formula, dataSensor0, nobs, ["hum", "co2"], "Response3") for nobs in n_obs]

    for itv_ in intervalLengths:
        formula = stlcg.Implies(subformula1=ϕHum, subformula2=stlcg.Eventually(stlcg.Always(ϕCo2, interval=[0,itv_]), interval=[0,60]))
        [runExperiment(formula, dataSensor0, nobs, ["hum", "co2"], f"Response3_{itv_}") for nobs in n_obs]

    print(True)