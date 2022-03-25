import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from matplotlib.ticker import MaxNLocator

# import the LinearRegression object
from sklearn.linear_model import LinearRegression

#plt.rc('text',usetex=True)
plt.rc('font',family='serif')

InitWeight = 250.0 #Weight on 12/31/2022, in lbs
GoalWeight = 200.0 #Desired goal weight

BasecCals = 2350.0 #Assumed base metabolic rate (the "2000 Calories" on the back of every snack)
GoalCals = 1350.0 #Goal for net calories per day

Dates = []
DateInts = []
Weights = []
FoodCals = []
ExerciseCals = []

with open("CaloriesData.txt") as f:
    lines = f.readlines()
    for line in lines:
        if (not "WWWWW" in line) and (not "Food" in line): #skip header and unfilled entries
            Weight = float( line.split()[2].replace('*','') )
            Date = line.split()[0]
            DateInt = int( line.split('    ')[1] )
            if "CCCC" in line or "EEE" in line:
                #assume that this incomplete data entry will match the prior averages
                FoodCal = np.median(np.array( FoodCals ))
                ExerciseCal = np.median(np.array( ExerciseCals ))
            else:
                FoodCal = float( line.split()[3] )
                ExerciseCal = float( line.split()[4].replace('\n','') )
            Dates.append(Date)
            DateInts.append(DateInt)
            Weights.append(Weight)
            FoodCals.append(FoodCal)
            ExerciseCals.append(ExerciseCal)

# make data into numpy arrays for convenience
Dates = np.array(Dates)
DateInts = np.array(DateInts)
Weights = np.array(Weights)
FoodCals = np.array(FoodCals)
ExerciseCals = np.array(ExerciseCals)
# get net calories
NetCals = FoodCals - ExerciseCals

#CurrentWeight = Weights[-1]
#CurrentWeight = np.mean(Weights[-5:])
CurrentWeight = min(Weights[-7:])
print()
print("---------- Analyzing "+str(len(DateInts))+" Days of Data ----------")
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


plt.bar(DateInts, NetCals+ExerciseCals, edgecolor='0.75', color='1.0', hatch='/////', label="Exercise")
for i in range(len(NetCals)):
    if NetCals[i] <= GoalCals:
        plt.bar(DateInts[i], NetCals[i], color='limegreen')
    else:
        plt.bar(DateInts[i], NetCals[i], color='red')
    if NetCals_RollingAvg[i] <= GoalCals:
        plt.scatter(DateInts[i], NetCals_RollingAvg[i], color='limegreen', s=7, zorder=50000)
    else:
        plt.scatter(DateInts[i], NetCals_RollingAvg[i], color='red', s=7, zorder=50000)
plt.scatter(DateInts, NetCals_RollingAvg, color='black', s=18, zorder=50000-1)
plt.plot(DateInts, NetCals_RollingAvg, color='black', linestyle='--', linewidth=1.0, zorder=50000-1, label="7-Day Rolling Avg.")
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
plt.xlim(min(DateInts)-1,max(DateInts)+0.99)
plt.legend(fontsize=10, loc='upper left', frameon=False)



plt.subplot(2, 2, 3)

x_fit = np.linspace(min(DateInts)-1,max(DateInts)+1)
plt.plot(x_fit, (BasecCals-GoalCals)*x_fit, linestyle='-', color='k', linewidth=1.0,label="Daily Calorie Goal", zorder=50000+1)

plt.bar(DateInts, CumulativeNetCals, color='0.5')

for i in range(len(NetCals)):
    if CumulativeNetCals[i] >= (i+1)*(BasecCals-GoalCals):
        plt.scatter(DateInts[i], CumulativeNetCals[i], color='limegreen', s=7, zorder=50000)
    else:
        plt.scatter(DateInts[i], CumulativeNetCals[i], color='red', s=7, zorder=50000)
plt.scatter(DateInts, CumulativeNetCals, color='0.5', s=18, zorder=50000-1)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("                                                                     Days Into 2022", fontsize=14)
plt.ylabel("Est. Total Calorie Deficit", fontsize=14)
plt.xlim(min(DateInts)-1,max(DateInts)+0.99)
plt.legend(fontsize=10, loc='upper left', frameon=False)


'''
# let's fit a simple linear regression
slr = LinearRegression(copy_X=True)
#add the initial point data point from Dec. 31, 2021 manually
slr.fit(np.insert(DateInts, 0, 0).reshape(-1,1), np.insert(Weights, 0, InitWeight))

dateints_fit = np.linspace(-0.5,max(DateInts)+0.5)
weights_fit = slr.intercept_ + dateints_fit*slr.coef_[0]
GoalDays = ( GoalWeight-slr.intercept_ ) / slr.coef_[0] - max(DateInts)

residuals = np.insert(Weights, 0, InitWeight) - slr.predict(np.insert(DateInts, 0, 0).reshape(-1,1))
residuals_std = np.std( residuals )
'''

#actually let's force the intercept to be InitWeight
slr = LinearRegression(copy_X=True, fit_intercept=False)
#add the initial point data point from Dec. 31, 2021 manually
slr.fit(np.insert(DateInts, 0, 0).reshape(-1,1), np.insert(Weights, 0, InitWeight)-InitWeight)
dateints_fit = np.linspace(-0.5,max(DateInts)+0.5)
weights_fit = InitWeight + dateints_fit*slr.coef_[0]
GoalDays = ( GoalWeight-InitWeight ) / slr.coef_[0] - max(DateInts)

residuals = np.insert(Weights, 0, InitWeight) - slr.predict(np.insert(DateInts, 0, 0).reshape(-1,1)) - InitWeight
residuals_std = np.std( residuals )


print()
print("So far you have been losing about "+str(round(-slr.coef_[0],2))+" pounds per day ("+str(round(-slr.coef_[0]*7,2))+" pounds per week).")

GoalDateObj = date.today() + timedelta(days=GoalDays)
GoalMonth = GoalDateObj.strftime("%B")
GoalDay = str(GoalDateObj.day)
print("If this trend holds, you will reach your goal weight in "+str(int(GoalDays))+" days (on "+GoalMonth+" "+GoalDay+").")

print()
print("You're doing great! Keep up the good work! :)")
print()


plt.subplot(2, 2, 2)

#plt.scatter(DateInts, Weights, color='k', marker='o', s=15)
#add the initial point data point from Dec. 31, 2021 manually
plt.scatter(np.insert(DateInts, 0, 0), np.insert(Weights, 0, InitWeight), color='k', marker='o', s=15)
plt.plot(dateints_fit, weights_fit, color='k', linestyle='-.', linewidth=1.5, label="Linear Fit")
#plt.axhline(GoalWeight, linestyle='-', color='gold', linewidth=1.5, label="Goal Weight")
plt.xticks([],fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel("Weight (lbs)", fontsize=14)
plt.xlim(min(DateInts)-1.5,max(DateInts)+0.5)
plt.gca().yaxis.set_ticks_position("right")
plt.gca().yaxis.set_label_position("right")
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend(fontsize=10, loc='upper right', frameon=False)


plt.subplot(2, 2, 4)

plt.text(max(DateInts), min(residuals), "Scatter $\sigma$ = "+str(round(residuals_std,2))+" lbs",
        ha='right', va='center', fontsize=10)

plt.scatter(np.insert(DateInts, 0, 0), residuals, color='k', marker='o', s=15)
plt.axhline(0, color='k', linestyle='-.', linewidth=1.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel("Fit Residuals (lbs)", fontsize=14)
plt.xlim(min(DateInts)-1.5,max(DateInts)+0.5)
plt.gca().yaxis.set_ticks_position("right")
plt.gca().yaxis.set_label_position("right")

plt.subplots_adjust(left=0.09, bottom=0.065, right=0.935, top=0.975, wspace=0, hspace=0)
plt.savefig('OutputImage.png',dpi=200)
plt.show()
