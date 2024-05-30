from opfunu.cec_based.cec2022 import *
import os
import re
from copy import deepcopy
import google.generativeai as genai

import warnings
warnings.filterwarnings("ignore")


genai.configure(api_key="Your Gemini API")
model = genai.GenerativeModel("gemini-pro")

PopSize = 100
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 30
MaxFEs = DimSize * 1000
curFEs = 0

MaxIter = int(MaxFEs / PopSize)
curIter = 0

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

FuncNum = 0

BestPop = None
BestFit = float("inf")


def Initialization(func):
    global Pop, FitPop, curFEs, DimSize, BestPop, BestFit
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func(Pop[i])
        curFEs += 1
    BestFit = min(FitPop)
    BestPop = deepcopy(Pop[np.argmin(FitPop)])


def DE(func):
    global Pop, FitPop, curIter, MaxIter, LB, UB, PopSize, DimSize, curFEs, BestPop, BestFit
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)
    for i in range(PopSize):
        CR = "Act as a parameter adaptor for differential evolution. "
        Insight = "The objective is to minimize the objective function. "
        Statement = "You need to determine the specific values for scaling factor F in (0, 2) and crossover rate Cr in (0, 1). The current best solution is " + str(
            BestPop) + " and the best objective value is " + str(BestFit) + ". "
        Experiment = "Give me the specific values of the scaling factor and crossover rate in the array-like format."

        try:
            response = model.generate_content(CR + Insight + Statement + Experiment)
            numbers = re.findall(r'\d+\.\d+', response.text)
            F, Cr = np.clip(float(numbers[0]), 0, 2), np.clip(float(numbers[1]), 0, 1)
        except Exception:
            F, Cr = np.random.uniform(0, 2), np.random.uniform(0, 1)

        r1, r2 = np.random.choice(list(range(PopSize)), 2, replace=False)
        Off[i] = BestPop + F * (Pop[r1] - Pop[r2])
        jrand = np.random.randint(0, DimSize)
        for j in range(DimSize):
            if np.random.rand() < Cr or j == jrand:
                pass
            else:
                Off[i][j] = Pop[i][j]
        Off[i] = np.clip(Off[i], LB, UB)
        FitOff[i] = func(Off[i])
    for i in range(PopSize):
        if FitOff[i] < FitPop[i]:
            Pop[i] = deepcopy(Off[i])
            FitPop[i] = FitOff[i]
            if FitOff[i] < BestFit:
                BestFit = FitOff[i]
                BestPop = deepcopy(Off[i])


def RunDE(func):
    global curFEs, curIter, MaxIter, MaxFEs, TrialRuns, Pop, FitPop, DimSize, FuncNum
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curFEs = 0
        curIter = 0
        np.random.seed(2024 + 88 * i)
        Initialization(func)
        Best_list.append(min(FitPop))
        while curIter < MaxIter:
            DE(func)
            curIter += 1
            Best_list.append(min(FitPop))
        All_Trial_Best.append(Best_list)
    np.savetxt("./DE_Data/CEC2022/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global FuncNum, DimSize, Pop, MaxFEs, MaxIter, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / PopSize)
    LB = [-100] * dim
    UB = [100] * dim

    CEC2022 = [F12022(Dim), F22022(Dim), F32022(Dim), F42022(Dim), F52022(Dim), F62022(Dim),
               F72022(Dim), F82022(Dim), F92022(Dim), F102022(Dim), F112022(Dim), F122022(Dim)]
    for i in range(len(CEC2022)):
        FuncNum = i + 1
        RunDE(CEC2022[i].evaluate)


if __name__ == "__main__":
    if os.path.exists('DE_Data/CEC2022') == False:
        os.makedirs('DE_Data/CEC2022')
    Dims = [20]
    for Dim in Dims:
        main(Dim)
