import openpyxl

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.linalg as la

import pandas as pd

file = 'BondsData.xlsx'
data = pd.ExcelFile(file)
print(data.sheet_names)
['Sheet1']

df = data.parse('Sheet1')
df.info
df.head(10)

ps = openpyxl.load_workbook('BondsData.xlsx')
sheet = ps['Sheet1']
print(sheet.max_row)

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
loc = ("documents")

bond1 = [0.75, 1.5, .5, 100.75, 100.75, 100.75, 100.75, 100.73, 100.72, 100.72, 100.71, 100.71, 100.71, 100.7, 100.7,
         100.7]
bond2 = [0.75, 1.5, 1, 101.44, 101.44, 101.41, 101.41, 101.40, 101.38, 101.38, 101.39, 101.38, 101.38, 101.38, 101.38]
bond3 = [.125, .25, 1.5, 100.2, 101.19, 100.16,
         100.16, 100.16, 100.14,
         100.16,
         100.17,
         100.17,
         100.18,
         100.19]
bond4 = [0.125, 0.25, 2,

         100.12,
         100.12,
         100.1,
         100.1,
         100.1,
         100.08,
         100.09,
         100.12,
         100.12,
         100.13,
         100.14,
         ]
bond5 = [0.25, 1.5, 2.33,
         103.13,
         103.14,
         103.1,
         103.09,
         103.09,
         103.04,
         103.06,
         103.1,
         103.08,
         103.06,
         103.09,
         ]
bond6 = [0.47, 2.25, 3.08,

         106.2,
         106.24,
         106.17,
         106.15,
         106.17,
         106.12,
         106.18,
         106.22,
         106.19,
         106.18,
         106.21,
         ]
bond7 = [0.06, 1.5, 3.58,

         104.25,
         104.28,
         104.25,
         104.19,
         104.24,
         104.29,
         104.26,
         104.32,
         104.24,
         104.24,
         104.27,
         ]
bond8 = [0.26, 1.25, 4.08,

         103.59,
         103.6,
         103.59,
         103.51,
         103.54,
         103.62,
         103.59,
         103.66,
         103.61,
         103.57,
         103.61,
         ]
bond9 = [0.02, 0.5, 4.58,

         100.32,
         100.32,
         100.3,
         100.22,
         100.27,
         100.37,
         100.33,
         100.42,
         100.35,
         100.33,
         100.37,
         ]
bond10 = [0.05, 0.25, 5.08,

          98.76,
          98.77,
          98.76,
          98.66,
          98.71,
          98.84,
          98.83,
          98.93,
          98.86,
          98.81,
          98.86,
          ]

Bgroup = [bond1, bond2, bond3, bond4, bond5, bond6, bond7, bond8, bond9, bond10]


def AppPrice(coupon, y, mat, face, freq):
    c = (coupon / 100) * face
    y = y / 100
    mat1 = mat // .5
    fP = 2 * (mat % 0.5)
    # FIX
    ai = (.5 - fP / 2) * (c)
    appPrice = ai
    fP = fP / 2
    appPrice = c / freq * (1 - (1 / (1 + y / freq) ** (mat1 + 2*fP))) / (y / freq) + face / ((1 + y / freq) ** (mat1 + 2*fP))
    return appPrice + ai


# this seems slightly off, but good enough for now (off by like 0.07?)
def AppPrice1(coupon, y, mat, face, freq):
    c = (coupon / 100) * face
    y = y / 100
    mat1 = int(mat // .5)
    fP = 2 * (mat % 0.5)
    # FIX
    ai = (.5 - fP / 2) * (c)
    appPrice = ai
    fP = fP / 2
    if fP > 0:
        for i in range(1, mat1 + 1):
            appPrice += (c / 2) * (1 / (1 + (y / 2)) ** (i + 2 * fP))

            if i == (mat1):
                appPrice += face * (1 / (1 + (y / 2)) ** (2 * (i / 2 + fP)))

    else:
        for i in range(1, mat1 + 1):
            appPrice += (c / 2) * (1 / (1 + y / 2) ** ((i)))

            if i == (mat1):
                appPrice += face * (1 / (1 + y / 2) ** ((i)))

    return appPrice


def DPrice(c, mat, face, freq, Price):
    c = (c / 100) * face
    mat1 = int(mat // .5)
    fP = mat % 0.5

    ai = (.5 - fP) * (c)
    return Price + ai


# Calculate YTM, account for dirty price
def CalcYTM(coupon, maturity, face, frequency, Price):
    H = 10
    L = 0
    DPrice1 = DPrice(coupon, maturity, face, frequency, Price)
    while (H - L) > 0.005:
        if AppPrice(coupon, (H + L) / 2, maturity, face, frequency) > DPrice1:
            L = (H + L) / 2
        else:
            H = (H + L) / 2

    return (H + L) / 2


# Need to figure out dirty prices
def YieldCurve(group):
    x = []
    y = [[], [], [], [], [], [], [], [], [], []]
    for i in group:
        x.append(i[2])

    for j in range(10):
        for i in group:
            y[j].append(CalcYTM(i[1], i[2], 100, 2, i[j + 3]))

    return [x, y]


# Spotrate, assume that bonds are properly spaced out
def DisRateDay(bonds, day):
    # assume we start from one payment
    ytm = []
    for i in range(len(bonds)):
        ytm.append(CalcYTM(bonds[i][1], bonds[i][2], 100, 2, bonds[i][day]))

    dis = [ytm[0]]
    for j in range(len(bonds)):
        cup = bonds[j][1] / 2
        tempsum = 0
        # sum up all terms
        # I think I sort of want to know about years, not sure how to do this thogu
        for k in range(j):
            tempsum += cup * (1 / (1 + ytm[j] / 2) ** (2 * bonds[k][2])) - cup * dis[k]

        tempsum = tempsum / cup
        dis.append(tempsum)

    return dis


def DisRate(ytm, sp):
    dis = []
    spot = []
    for j in range(len(ytm)):

        tempsum = 0
        # sum up all terms
        # I think I sort of want to know about years, not sure how to do this thogu

        if sp > 0:
            for k in range(j+1):
                tempsum += (1 / (1 + ytm[j] / 2) ** (2 * ((k) / 2 + sp)))
            for k in range(0, j):
                tempsum -= dis[k]
            tempsum = abs(tempsum)
            dis.append(tempsum)

            ex = (2 * ((j + 1) / 2 + sp))
            nspot = 2 * tempsum ** (-1 / ex) - 2
            spot.append(nspot)


        else:
            for k in range(0, j + 1):
                tempsum += (1 / (1 + ytm[j] / 2)) ** (2 * ((k+1) / 2))
            for k in range(0, j):
                tempsum -= dis[k]
            tempsum = abs(tempsum)
            dis.append(tempsum)

            ex = (2 * ((j + 1) / 2))
            nspot = 2 * tempsum ** (-1 / ex) - 2
            spot.append(nspot)

    return [dis, spot]


def Spot(bonds, ytm):
    dis = DisRate(bonds, day)
    spot = []
    for j in range(len(bonds)):
        if dis[j] == 0:
            spot.append(0)
        else:
            sj = 2 * (abs(dis[j])) ** (-1 / 2 * bonds[j][2]) - 2
            spot.append(sj)
    return spot

def ForwardRates(Spots):
    Forwards = []
    for i in range(len(Spots) - 1):
        nf = (((1+ Spots[i+1]/2)**(2*(i+1))) / ((1+ Spots[i]/2)**(2*(i)))) - 1
        Forwards.append(nf)

    return Forwards

def CalcLogRates(Yields):
    logR = []
    for i in range(len(Yields)-1):
        l = np.log(Yields[i+1]) - np.log(Yields[i])
        logR.append(l)
    return logR


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def cov(a, b):

    if len(a) != len(b):
        return

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    sum = 0

    for i in range(0, len(a)):
        sum += ((a[i] - a_mean) * (b[i] - b_mean))

    return sum/(len(a))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #All the bonds data
    bond1 = [0.75, 1.5, .5, 100.75, 100.75, 100.75, 100.75, 100.73, 100.72, 100.72, 100.71, 100.71, 100.71, 100.7,
             100.7, 100.7]
    bond2 = [0.75, 1.5, 1, 101.44, 101.44, 101.41, 101.41, 101.40, 101.38, 101.38, 101.39, 101.38, 101.38, 101.38,
             101.38]
    bond3 = [.125, .25, 1.5, 100.2, 101.19, 100.16,
             100.16, 100.16, 100.14,
             100.16,
             100.17,
             100.17,
             100.18,
             100.19]
    bond4 = [0.125, 0.25, 2,

             100.12,
             100.12,
             100.1,
             100.1,
             100.1,
             100.08,
             100.09,
             100.12,
             100.12,
             100.13,
             100.14,
             ]
    bond5 = [0.25, 1.5, 2.33,
             103.13,
             103.14,
             103.1,
             103.09,
             103.09,
             103.04,
             103.06,
             103.1,
             103.08,
             103.06,
             103.09,
             ]
    bond6 = [0.47, 2.25, 3.08,

             106.2,
             106.24,
             106.17,
             106.15,
             106.17,
             106.12,
             106.18,
             106.22,
             106.19,
             106.18,
             106.21,
             ]
    bond7 = [0.06, 1.5, 3.58,

             104.25,
             104.28,
             104.25,
             104.19,
             104.24,
             104.29,
             104.26,
             104.32,
             104.24,
             104.24,
             104.27,
             ]
    bond8 = [0, 1.25, 4.08,

             103.59,
             103.6,
             103.59,
             103.51,
             103.54,
             103.62,
             103.59,
             103.66,
             103.61,
             103.57,
             103.61,
             ]
    bond9 = [0.02, 0.5, 4.58,

             100.32,
             100.32,
             100.3,
             100.22,
             100.27,
             100.37,
             100.33,
             100.42,
             100.35,
             100.33,
             100.37,
             ]
    bond10 = [0.05, 0.25, 5.08,

              98.76,
              98.77,
              98.76,
              98.66,
              98.71,
              98.84,
              98.83,
              98.93,
              98.86,
              98.81,
              98.86,
              ]

    Bgroup = [bond1, bond2, bond3, bond4, bond5, bond6, bond7, bond8, bond9, bond10]


    data = YieldCurve(Bgroup)
    x = data[0]
    x.insert(0, 0)
    print(x)
    y1 = data[1][0]
    y2 = data[1][1]
    y3 = data[1][2]
    y4 = data[1][3]
    y5 = data[1][4]
    y6 = data[1][5]
    y7 = data[1][6]
    y8 = data[1][7]
    y9 = data[1][8]
    y10 = data[1][9]

    print(data[1][8])
    #bond yields from website
    ytm = [.08, .13, .14, .2, .21, .24, .35, .41, .48, .54]
    #splines = interpolate.splrep(x, ytm)

    #4a)
    for i in range(10):
        y = data[1][i]
        z = y[:]
        z.insert(0, 0)
        splines = interpolate.splrep(x, z)
        x_vals = np.linspace(0, 5)
        y_vals = interpolate.splev(x_vals, splines)
        #plt.plot(x, z, 'o', label='YTM for day ' +str(i+1))
        #plt.plot(x_vals, y_vals, '-x', label= 'YTM curve for day ' + str(i+1))
        #plt.xlabel('Time to Maturity')
        #plt.ylabel('Yield to Maturity')
        #plt.title('YTM Curve')


   






    #4b)
    print('testing spots')

    Aspots = []
    for i in range(10):
        # spots
        # first 4 spots
        y = data[1][i]
        z = y[:]
        z.insert(0,0)
        y1 = z[:5]
        spots = DisRate(y1, 0)[1]
        print('first spot')
        print(spots)
        # Next spot

        #inteporsate ytm to get next spot 4b)
        ytms2 = []
        print('TESTING1')
        print(y1)
        print(z)
        splines = interpolate.splrep(x, z)
        for j in range(5):
            tmp = (interpolate.splev(j / 2 + 0.33, splines))
            tmp = float(tmp)
            ytms2.append(tmp)
        print("TESTING")
        print(ytms2)
        spot5 = DisRate(ytms2, 0.33)[1][4]
        print(spot5)
        spots.append(spot5)


        #nextspot
        print('LAST SPOTS')
        ytms3 = []

        for k in range(11):
            tmp = (interpolate.splev(k / 2 + 0.08, splines))
            tmp = float(tmp)
            ytms3.append(tmp)

        print(ytms3)
        spot6 = DisRate(ytms3, 0.08)[1]
        print(spot6)
        spots += spot6[6:]
        Aspots.append(spots)
        print(spots)


        splines = interpolate.splrep(x, spots)
        x_vals = np.linspace(0, 5)
        y_vals = interpolate.splev(x_vals, splines)
        #plt.plot(x, spots, 'o', label='Spot Rates for day ' + str(i + 1))
        #plt.plot(x_vals, y_vals, '-x', label='Spot curve for day ' + str(i + 1))
        #plt.xlabel('Time to Maturity')
        #plt.ylabel('Spot Rate')
        #plt.title('Spot Curve')

    #plt.legend()
    #plt.show()
#4c)
    #interpolate between spots

    AFR = []
    for j in range(10):
        Newspots = []
        print(Aspots[1])
        splines2 = interpolate.splrep(x, Aspots[j])
        for i in range(5):
            Newspots.append(interpolate.splev(i+1, splines2))

        FR = ForwardRates(Newspots)
        AFR.append(FR)
        splines = interpolate.splrep([1,2,3,4], FR)
        x_vals = np.linspace(1, 4)
        y_vals = interpolate.splev(x_vals, splines)
        #plt.plot([1, 2, 3, 4], FR, 'o', label='Forward rates for day ' + str(j+1))
        #plt.plot(x_vals, y_vals, '-x', label='Forward curve for day ' + str(j + 1))
        #plt.xlabel('Time to Maturity')
        #plt.ylabel('Forward Rates')
        #plt.title('Forward Curve')
    #plt.legend()
    #plt.show()


#5)
    #Get random variables
    #Yields COV
    ytmY = []
    for k in range(5):
        #Get a yield curve
        yields = []
        for j in range(10):
            y = data[1][j]
            z = y[:]
            z.insert(0, 0)
            splines = interpolate.splrep(x, z)
            rkj = interpolate.splev(k+1, splines)
            yields.append(float(rkj))
        ytmY.append(yields)

    print(ytmY)

    Rvs = []
    for i in range(len(ytmY)):
        rows = []
        for j in range(len(ytmY[i])-1):
            np.log(ytmY[i][j+1]) - np.log(ytmY[i][j])
            rows.append(np.log(ytmY[i][j+1]) - np.log(ytmY[i][j]))
        Rvs.append(rows)

    Cov = []
    print(len(Rvs))
    print(len(Rvs[1]))
    for i in range(len(Rvs)):
        row = []
        for j in range(len(Rvs)):
            row.append(cov(Rvs[i], Rvs[j]))

        Cov.append(row)

    Cov = np.array(Cov)
    print('COVVVVV')
    print(Cov)
    print('DONE')

    #futureRates


    #print(AFR2)
    AFR2 = []
    for i in range(len(AFR[1])):
        tmp = []
        for j in range(len(AFR)):
            tmp.append(AFR[j][i])
        AFR2.append(tmp)
    #print(AFR2)
    print(len(AFR2[1]))
    #print(AFR2)
    print(AFR2)

    Rvs2 = []
    #for i in range(len(AFR2)):
        #rows = []
        #for j in range(len(AFR2[i]) - 1):
            #np.log(AFR2[i][j + 1]) - np.log(AFR2[i][j])
            #rows.append(np.log(AFR2[i][j + 1]) - np.log(AFR2[i][j]))
        #Rvs2.append(rows)

    Cov2 = []
    print('HI')
    print(len(Rvs2))


    #for i in range(len(Rvs2)):
        #row = []
        #for j in range(len(Rvs2)):
            #row.append(cov(Rvs2[i], Rvs2[j]))

        #Cov2.append(row)


    #print(Cov2)

    #6
    eigvecs1, eigvals1 = la.eig(Cov)
    print("Eigenvectors:")
    print(eigvecs1)
    print("Eigenvalues:")
    print(eigvals1)


    #eigvals2, eigvecs2 = la.eig(Cov2)









    #for i in range(len(AFR2)):
        #row = []
        #for j in range(len(AFR2)):
            #row.append(cov(AFR2[i], AFR2[j]))

        #Cov2.append(row)

    #Cov2 = np.array(Cov2)
    #print('BREAK')
    #print(Cov2)
    #print(Aspots)

















    #splines1 = interpolate.splrep(x, y)
    # get interpolation values (MIGHT WANT TO AVOID 0 ISSUE)
    #tempspots = []
    #for i in range(4):
        #tmp = (interpolate.splev(i / 2 + 0.33, splines1))
        #print(tmp)
        #tempspots.append(tmp)

    #print(tempspots)
    #print('HELLO!')

    # now want to interpolate to 2.33

    # ytm interpolation
    x = data[0]
    #print(x)
    #print(ytm)
    y = data[1][9]
    #splines = interpolate.splrep(x, ytm)
    #x_vals = np.linspace(0, 5)
    #y_vals = interpolate.splev(x_vals, splines)
    #plt.plot(x, ytm, 'o')
    #plt.plot(x_vals, y_vals, '-x')
    # plt.show()
    #print(y_vals)
    #print(splines)
    # plt.plot(x, y)
    # plt.show()
    # print(data)
    #print(y)

    dis = DisRateDay(Bgroup, 4)
    #print(dis)
    # print(DisRateDay(Bgroup, 4))
    # print(Spot(Bgroup, 5))

    # print(y)
    # tck = interpolate.splrep(x, y)
    # f = interpolate.splev(x, tck)

    # plt.figure()
    # plt.plot(x, y, x, f)
    # print(f)
    # plt.show()

    #

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
