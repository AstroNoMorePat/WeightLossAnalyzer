import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

# import the LinearRegression object
from sklearn.linear_model import LinearRegression

#plt.rc('text',usetex=True)
plt.rc('font',family='serif')

InitDate = date(2021,12,31)

InitWeight = 250.0 #Weight on 12/31/2021, in lbs
GoalWeight = 200.0 #Desired goal weight

BasecCals = 2350.0 #Assumed base metabolic rate (the "2000 Calories" on the back of every snack)
GoalCals = 1500.0 #Goal for net calories per day

DateIntsW = []
Weights = []
DateIntsC = []
FoodCals = []
ExerciseCals = []

with open("CaloriesData.txt") as f:
    lines = f.readlines()
    for line in lines[1:]:
        if not ("WWWWW" in line and "CCCC" in line): #skip completely unfilled entries
            if "WWWWW" in line:
                DateInt = int( line.split('    ')[1] )
                FoodCal = float( line.split()[3] )
                ExerciseCal = float( line.split()[4].replace('\n','') )
                DateIntsC.append(DateInt)
                FoodCals.append(FoodCal)
                ExerciseCals.append(ExerciseCal)
            elif "CCCC" in line:
                #assume that this incomplete data entry will match the prior medians
                Weight = float( line.split()[2])
                #FoodCal = np.median(np.array( FoodCals ))
                #ExerciseCal = np.median(np.array( ExerciseCals ))
                DateInt = int( line.split('    ')[1] )
                DateIntsW.append(DateInt)
                Weights.append(Weight)
                #DateIntsC.append(DateInt)
                #FoodCals.append(FoodCal)
                #ExerciseCals.append(ExerciseCal)
            else:
                #normal days with full weight and calorie data
                Weight = float( line.split()[2])
                DateInt = int( line.split('    ')[1] )
                FoodCal = float( line.split()[3] )
                ExerciseCal = float( line.split()[4].replace('\n','') )
                DateIntsW.append(DateInt)
                Weights.append(Weight)
                DateIntsC.append(DateInt)
                FoodCals.append(FoodCal)
                ExerciseCals.append(ExerciseCal)

# make data into numpy arrays for convenience
DateIntsW = np.array(DateIntsW)
DateIntsC = np.array(DateIntsC)
Weights = np.array(Weights)
FoodCals = np.array(FoodCals)
ExerciseCals = np.array(ExerciseCals)
# get net calories
NetCals = FoodCals - ExerciseCals

CurrentDate = InitDate + timedelta(days=float(max(DateIntsC)))
CurrentMonth = CurrentDate.strftime("%B")
CurrentDay = str(CurrentDate.day)
CurrentYear = str(CurrentDate.year)

#CurrentWeight = Weights[-1]
#CurrentWeight = np.mean(Weights[-5:])
CurrentWeight = min(Weights[-7:])
print()
print("---------- Analyzing "+str(len(DateIntsC))+" Days of Data ----------")
print("Current Date: "+CurrentMonth+" "+CurrentDay+", "+CurrentYear)
print()
print("Starting Weight: "+str(InitWeight)+" lbs")
print("Current Weight:  "+str(round(CurrentWeight,1))+" lbs")
print("Goal Weight:     "+str(GoalWeight)+" lbs")
print()
print("Way to go! You're "+str(round(100.0*(InitWeight-CurrentWeight)/(InitWeight-GoalWeight),1))+"% of the way to your goal!")

print()
print("Average Daily Calories Eaten (All of 2022): "+str(round(np.mean(FoodCals),1)))
print("Average Daily Net Calories (All of 2022): "+str(round(np.mean(NetCals),1)))
if np.mean(NetCals) >= GoalCals:
    print("Net Calories Above Goal (All of 2022): "+str(int(np.sum(NetCals)-GoalCals*len(NetCals))))
elif np.mean(NetCals) < GoalCals:
    print("Net Calories Below Goal (All of 2022): "+str(int(GoalCals*len(NetCals)-np.sum(NetCals))))
print()
print("Average Daily Calories Eaten (Past Week): "+str(round(np.mean(FoodCals[-7:]),1)))
print("Average Daily Net Calories (Past Week): "+str(round(np.mean(NetCals[-7:]),1)))
if np.mean(NetCals[-7:]) >= GoalCals:
    print("Net Calories Above Goal (Past Week): "+str(int(np.sum(NetCals[-7:])-7*GoalCals)))
elif np.mean(NetCals[-7:]) < GoalCals:
    print("Net Calories Below Goal (Past Week): "+str(int(7*GoalCals-np.sum(NetCals[-7:]))))
print()

TotExerciseCals = np.sum(ExerciseCals)
print("Total Exercise Calories: "+str(int(TotExerciseCals)))
print("This is equivalent to burning about "+str(round(TotExerciseCals/3500.0,1))+" lbs of fat. Nicely done!")
print()


# now look at cumulative calories relative to the assumed base metabolic rate
CumulativeNetCals = []
for i in range(len(NetCals)):
    if len(CumulativeNetCals) < 1:
        CumulativeNetCals.append( NetCals[i]-BasecCals )
    else:
        CumulativeNetCals.append( CumulativeNetCals[i-1] + (NetCals[i]-BasecCals) )
# this is essentially the total calorie deficit since starting the diet, so we'll adopt a negative convention
CumulativeNetCals = -np.array(CumulativeNetCals)

LostWeight = InitWeight-CurrentWeight
print("Total Calorie Deficit (Assuming "+str(int(BasecCals))+" Cal/Day Base Metabolism): "+str(int(CumulativeNetCals[-1])))
print("Expected Weight Loss: About "+str(round(CumulativeNetCals[-1]/3500.0,1))+" lbs")
print("Actual Weight Loss: "+str(round(LostWeight,1))+" lbs")

TrueBaseCal = (LostWeight - CumulativeNetCals[-1]/3500.0)*3500.0/len(Weights) + BasecCals
print("This implies a true base metabollic rate of about "+str(int(TrueBaseCal))+" calories per day")

#get a rolling 7-day average for NetCals
NetCals_RollingAvg = []
for i in range(len(NetCals)):
    if i >= 6:
        #print(i, NetCals[i-6:i+1])
        NetCals_RollingAvg.append(np.mean(NetCals[i-6:i+1]))
    else:
        #print(i, NetCals[:i+1])
        NetCals_RollingAvg.append(np.mean(NetCals[:i+1]))

plt.figure(figsize=(10,8.0))


plt.subplot(2, 2, 1)

cscale = 3
bkgscale = 8

#plt.bar(DateIntsC, NetCals+ExerciseCals, edgecolor='0.75', color='1.0', hatch='/////', label="Exercise", width=1.0)
plt.bar(DateIntsC, NetCals+ExerciseCals, color='0.75', label="Exercise", width=1.0)
for i in range(len(NetCals)):
    if NetCals[i] <= GoalCals:
        plt.bar(DateIntsC[i], NetCals[i], color='limegreen', width=1.0,alpha=0.7)
    else:
        plt.bar(DateIntsC[i], NetCals[i], color='red', width=1.0,alpha=0.7)
    if NetCals_RollingAvg[i] <= GoalCals:
        plt.scatter(DateIntsC[i], NetCals_RollingAvg[i], color='limegreen', s=cscale, zorder=50000)
    else:
        plt.scatter(DateIntsC[i], NetCals_RollingAvg[i], color='red', s=cscale, zorder=50000)
plt.scatter(DateIntsC, NetCals_RollingAvg, color='black', s=bkgscale, zorder=50000-1)
plt.plot(DateIntsC, NetCals_RollingAvg, color='black', linestyle='--', linewidth=1.0, zorder=50000-1, label="7-Day Rolling Avg.")
#if np.mean(NetCals[-7:]) <= GoalCals:
#    plt.axhline(np.mean(NetCals[-7:]), linestyle=':', color='limegreen', linewidth=1.0, label="Average Calories (Past Week)")
#else:
#    plt.axhline(np.mean(NetCals[-7:]), linestyle=':', color='red', linewidth=1.0, label="Average Calories (Past Week)")
#if np.mean(NetCals) <= GoalCals:
#    plt.axhline(np.mean(NetCals), linestyle='-.', color='limegreen', linewidth=1.0, label="Average Calories (Total)")
#else:
#    plt.axhline(np.mean(NetCals), linestyle='-.', color='red', linewidth=1.0, label="Average Calories (Total)")
plt.axhline(GoalCals, linestyle='-', color='k', linewidth=1.0,label="Daily Calorie Goal")
plt.xticks([],fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel("Daily Net Calories", fontsize=14)
plt.ylim(0,4200)
plt.xlim(min(DateIntsC)-1,max(DateIntsC)+0.99)
plt.legend(fontsize=10, loc='upper left', frameon=False)



plt.subplot(2, 2, 3)

x_fit = np.linspace(min(DateIntsC)-1,max(DateIntsC)+1)
plt.plot(x_fit, (BasecCals-GoalCals)*x_fit, linestyle='-', color='k', linewidth=1.0,label="Daily Calorie Goal", zorder=50000+1)

plt.bar(DateIntsC, CumulativeNetCals, color='0.75', width=1.0)

for i in range(len(NetCals)):
    if CumulativeNetCals[i] >= (i+1)*(BasecCals-GoalCals):
        plt.scatter(DateIntsC[i], CumulativeNetCals[i], color='limegreen', s=cscale, zorder=50000)
    else:
        plt.scatter(DateIntsC[i], CumulativeNetCals[i], color='red', s=cscale, zorder=50000)
plt.scatter(DateIntsC, CumulativeNetCals, color='k', s=bkgscale, zorder=50000-1)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("                                                                     Days Into 2022", fontsize=14)
plt.ylabel("Est. Total Calorie Deficit", fontsize=14)
plt.xlim(min(DateIntsC)-1,max(DateIntsC)+0.99)
plt.legend(fontsize=10, loc='upper left', frameon=False)



loss_rate = (InitWeight-CurrentWeight) / max(DateIntsC)
print()
print("So far you have been losing about "+str(round(loss_rate,2))+" pounds per day ("+str(round(7.0*loss_rate,2))+" pounds per week).")

GoalDays = ( CurrentWeight - GoalWeight ) / loss_rate
#GoalDateObj = date.today() + timedelta(days=GoalDays)
GoalDateObj = InitDate + timedelta(days=float(max(DateIntsC))) + timedelta(days=GoalDays)
GoalMonth = GoalDateObj.strftime("%B")
GoalDay = str(GoalDateObj.day)
print("If this trend holds, you will reach your goal weight in "+str(int(GoalDays))+" days (on "+GoalMonth+" "+GoalDay+").")


# let's fit a simple linear regression
slr = LinearRegression(copy_X=True)
#add the initial point data point from Dec. 31, 2021 manually
slr.fit(np.insert(DateIntsW, 0, 0).reshape(-1,1), np.insert(Weights, 0, InitWeight))

dateints_fit = np.linspace(-0.5,max(DateIntsW)+0.5)
weights_fit = slr.intercept_ + dateints_fit*slr.coef_[0]
#GoalDays = ( GoalWeight-slr.intercept_ ) / slr.coef_[0] - max(DateIntsW)

#residuals = np.insert(Weights, 0, InitWeight) - slr.predict(np.insert(DateIntsW, 0, 0).reshape(-1,1))
#residuals_std = np.std( residuals )


def multi_component_line(x, m1, b1, t1, m2, t2, m3):
    y_out = []
    for x_in in x:
        if x_in < t1:
            y = m1*x_in + b1
        elif x_in >= t1 and x_in < t2:
            y = m2*(x_in-t1) + m1*t1 + b1
        elif x_in >= t2:
            y = m3*(x_in-t2) + m2*(t2-t1) + m1*t1 + b1
        y_out.append(y)
    return y_out

mb  = [-0.5 , -0.00001]
bb  = [50.0 , 500.0]
t1b = [20.0 ,  60.0]
t2b = [85.0 , 155.0]
param_bounds = ((mb[0],bb[0],t1b[0],mb[0],t2b[0],mb[0]),(mb[1],bb[1],t1b[1],mb[1],t2b[1],mb[1]))
init_guess = [-0.2,250.0,35.0,-0.15,100.0,-0.1]

popt, pcov = curve_fit(multi_component_line, DateIntsW, Weights, p0=init_guess, bounds=param_bounds)

m1_fit = popt[0]
b1_fit = popt[1]
t1_fit = popt[2]
m2_fit = popt[3]
t2_fit = popt[4]
m3_fit = popt[5]


multi_component_fit = multi_component_line(DateIntsW, m1_fit, b1_fit, t1_fit, m2_fit, t2_fit, m3_fit)

residuals = np.insert(Weights, 0, InitWeight) - multi_component_line(np.insert(DateIntsW, 0, 0), m1_fit, b1_fit, t1_fit, m2_fit, t2_fit, m3_fit)
residuals_std = np.std( residuals )

GoalDays = (GoalWeight-m2_fit*(t2_fit-t1_fit)-m1_fit*t1_fit-b1_fit)/m3_fit + t2_fit - max(DateIntsW)

#GoalDateObj = date.today() + timedelta(days=GoalDays)
GoalDateObj = InitDate + timedelta(days=float(max(DateIntsC))) + timedelta(days=GoalDays)
GoalMonth = GoalDateObj.strftime("%B")
GoalDay = str(GoalDateObj.day)
GoalYear = str(GoalDateObj.year)

print()
print("Multi-Component Fit:")
print("From day 0 to day "+str(int(t1_fit))+", the loss rate is "+str(round(-m1_fit,3))+" lbs/day ("+str(round(-7*m1_fit,2))+" lbs/week)")
print("From day "+str(int(t1_fit))+" to day "+str(int(t2_fit))+", the loss rate is "+str(round(-m2_fit,3))+" lbs/day ("+str(round(-7*m2_fit,2))+" lbs/week)")
print("From day "+str(int(t2_fit))+" onward, the loss rate is "+str(round(-m3_fit,3))+" lbs/day ("+str(round(-7*m3_fit,2))+" lbs/week)")

print()
if GoalDays + max(DateIntsW) > 365.0:
    print("Based instead on an extrapolation of the multi-component fit described above,"+"\n"+"you will reach your goal weight in "+str(int(GoalDays))+" days (on "+GoalMonth+" "+GoalDay+", "+GoalYear+").")
else:
    print("Based instead on an extrapolation of the multi-component fit described above,"+"\n"+"you will reach your goal weight in "+str(int(GoalDays))+" days (on "+GoalMonth+" "+GoalDay+").")

print()
print("You're doing great! Keep up the good work! :)")
print()


plt.subplot(2, 2, 2)

wscale = 8


#plt.scatter(DateIntsW, Weights, color='k', marker='o', s=15)
#add the initial point data point from Dec. 31, 2021 manually
plt.scatter(np.insert(DateIntsW, 0, 0), np.insert(Weights, 0, InitWeight), color='k', marker='o', s=wscale)
plt.plot(dateints_fit, weights_fit, color='k', linestyle=':', linewidth=1.5, label="Linear Fit")
plt.plot(DateIntsW, multi_component_fit, color='k', linestyle='-.', linewidth=1.5, label="Multi-Component Fit")
#plt.axhline(GoalWeight, linestyle='-', color='gold', linewidth=1.5, label="Goal Weight")
plt.xticks([],fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel("Weight (lbs)", fontsize=14)
plt.xlim(min(DateIntsW)-2.5,max(DateIntsW)+2.5)
plt.gca().yaxis.set_ticks_position("right")
plt.gca().yaxis.set_label_position("right")
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend(fontsize=10, loc='upper right', frameon=False)


plt.subplot(2, 2, 4)

plt.text(min(DateIntsW), max(residuals),
        "Scatter $\sigma$ = "+str(round(residuals_std,2))+" lbs",
        ha='left', va='center', fontsize=10)

plt.scatter(np.insert(DateIntsW, 0, 0), residuals, color='k', marker='o', s=wscale)
plt.axhline(0, color='k', linestyle='-.', linewidth=1.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel("Fit Residuals (lbs)", fontsize=14)
plt.xlim(min(DateIntsW)-2.5,max(DateIntsW)+2.5)
plt.gca().yaxis.set_ticks_position("right")
plt.gca().yaxis.set_label_position("right")

plt.subplots_adjust(left=0.095, bottom=0.065, right=0.935, top=0.975, wspace=0, hspace=0)
plt.savefig('OutputImage.png',dpi=200)
plt.show()
