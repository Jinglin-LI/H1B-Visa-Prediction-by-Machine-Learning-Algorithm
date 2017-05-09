import numpy as np
import scipy as sci
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from math import isnan
from sklearn import tree
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)


class H1bInfo():


    #load the data from path
    def read_process_data(self,dataPath):
        H1Info = pd.read_csv(dataPath)
        return H1Info


    #Analyze the H1Bs by the status of their visa applications and show the plot
    def showCASE_STATUS(self,H1Info):
        statusCount = H1Info['CASE_STATUS'].value_counts()
        statusTypes = statusCount.index.copy(deep=True)
        statusTypes = statusTypes.values
        statusTypes[4] = 'Others'   # PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED , simplified to 'other'
        statusTypes = statusTypes[0:5]

        statusValues = statusCount.copy()
        statusValues = statusValues.values
        statusValues[4] = np.sum(statusValues[4:7])  # sum all of the others
        statusValues = statusValues[0:5]


        piefig = plt.pie(statusValues, autopct='%1.00f%%', shadow=False, startangle=90)
        plt.legend(statusTypes, loc="best", prop={'size':6})
        plt.axis('equal')
        plt.tight_layout()
        plt.title('Case Status percentage')
        plt.show()


    #K-Means Clustering to seperate the h1b location
    def K_meanAnalyze(self,H1Info):
        #Get the states values
        H1City,H1State = H1Info['WORKSITE'].str.split(', ',1).str
        #Reading Latitude and Longitude data (taking out null)
        H1LatLong = H1Info.loc[H1State != 'NA']
        H1LatLong = H1LatLong[['lon','lat']]
        H1LatLong = H1LatLong.dropna()
        H1Long = H1LatLong['lon'].values
        H1Lat = H1LatLong['lat'].values

        #K-Means clustering to see where the applicants were
        nClusters = 10
        kmeans = KMeans(n_clusters=nClusters, random_state=0).fit(H1LatLong.values)
        #Getting the cluster for each case
        kl = kmeans.labels_
        H1LatLong['Cluster'] = kl
        H1LatLong['State'] = H1State
        H1LatLong = H1LatLong.dropna()
        return H1LatLong

    #show the plot of the h1b WORKSITE after k-mean
    #dense is the parameter to control distance between point, when gainning the info from the H1LatLong
    def showWORKSITE(self,dense,H1LatLong):


        #dense = 2000/1000 (usually )
        pltIndex = range(0,2892144,dense)

        #plot collor
        colorSet = ['ro','go','bo','co','mo','ko','r*','g*','b*','c*']
        CIndex = range(0,10)

        for i in CIndex:
            plt.plot(H1LatLong.iloc[pltIndex,0].loc[H1LatLong.iloc[pltIndex,2] == i], H1LatLong.iloc[pltIndex,1].loc[H1LatLong.iloc[pltIndex,2] == i], colorSet[i])

        #axis label name
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend(CIndex, loc="upper left", prop={'size':12})
        plt.title('H1b application distribution')
        plt.show()

    #Salary Data according to clusters
    def salaryAnalyze(self,H1Info,H1LatLong):

        nClusters = 10
        H1Salary = H1LatLong
        H1Salary['Salary'] = H1Info['PREVAILING_WAGE']
        H1Salary['FT'] = H1Info['FULL_TIME_POSITION']

        salaryMin = np.zeros(nClusters)
        salaryMax = np.zeros(nClusters)
        salaryMean = np.zeros(nClusters)
        salaryMedian = np.zeros(nClusters)
        salaryStd = np.zeros(nClusters)

        nClusters = 10
        for k in range(0,nClusters):
            #which cluster belongs to
            salaryClustered = H1Salary.loc[H1Salary['Cluster'] == k]
            #only consider full-time employment
            salaryClustered = salaryClustered.loc[salaryClustered['FT'] == 'Y']
            #take out extreme outliers
            salaryClustered = salaryClustered.loc[salaryClustered['Salary'] <= 5000000]
            # take out non-paid positions
            salaryClustered = salaryClustered.loc[salaryClustered['Salary'] > 0]
            #drop N/A
            salaryClustered = salaryClustered.dropna()

            #each cluster's description
            salaryMin[k] = np.min(salaryClustered['Salary'].values)
            salaryMax[k] = np.max(salaryClustered['Salary'].values)
            salaryMean[k] = np.mean(salaryClustered['Salary'].values)
            salaryMedian[k] = np.median(salaryClustered['Salary'].values)
            salaryStd[k] = np.std(salaryClustered['Salary'].values)
        return [salaryMin,salaryMax,salaryMean,salaryMedian,salaryStd]

    #show the salary detail after k-mean process
    def showSALARY_table(self,salaryMin,salaryMax,salaryMean,salaryMedian,salaryStd):
         #build a DataFrame to show.
        salaryStats = pd.DataFrame()
        salaryStats['Mean'] = salaryMean.astype(int)
        salaryStats['Median'] = salaryMedian.astype(int)
        salaryStats['Std'] = salaryStd.astype(int)
        salaryStats['Min'] = salaryMin.astype(int)
        salaryStats['Max'] = salaryMax

        print salaryStats

    #Plotting and comparison to the median US salary (2015)

    def showSALARY_plot(self,salaryMedian,salaryMean):
        colorSet = ['r','g','b','c','m','k','r','g','b','c']
        colorSetS = ['r*','g*','b*','c*','m*','k*','r*','g*','b*','c*']
        CIndex = range(0,10)
        for i in CIndex:
            plt.bar(i,salaryMedian[i], color = colorSet[i])

        for i in CIndex:
            plt.plot(i,salaryMean[i], colorSet[i])

        plt.plot(np.arange(-0.5,10.5,1),55775*np.ones(11), "b--")
        plt.xlabel('Cluster number.')
        plt.ylabel('Median Salary (USD)')
        plt.title('Salary for every work part')
        plt.show()

    #show top 6 company who has apply to the h1b for employee
    def showTOP6com_table(self,H1Info):
        comptable = H1Info['EMPLOYER_NAME'].value_counts().sort_values(ascending=False).head(6)
        comptable.plot(kind='bar')
        plt.ylabel('H1B number')
        plt.xlabel('company_name')
        plt.title('Top6 company application number')

        comptable2=H1Info[H1Info['EMPLOYER_NAME'].isin(comptable.index.values)]
        comptable2 = comptable2.groupby(['EMPLOYER_NAME','YEAR']).size().unstack()
        comptable2.plot(kind='bar')
        plt.ylabel('H1B number')
        plt.xlabel('company_name')
        plt.legend(loc="upper right", prop={'size':8})
        plt.title('Company history application number')
        plt.show()

    #show the h1b number in everyear's change
    def showYearTrend_plot(self,H1Info):
        yearTrend = H1Info['YEAR'].value_counts().sort_values(ascending=True)
        yearTrend.plot(kind = 'bar')
        plt.title('Recent year for H1B application number')
        plt.show()

    #show top20 popular Jobtitle and top10 Worksites for H1-B Visa holders
    def showJOBTITLE_plot(self,H1Info):
        H1Info['JOB_TITLE'].value_counts().sort_values(ascending=False).head(20).plot(kind='barh',color=tableau20)
        plt.title('Top10 jobs for h1b application')
        plt.show()
        H1Info['WORKSITE'].value_counts().head(10).plot(kind='barh',color=tableau20)
        plt.title('Top10 cities for h1b application')
        plt.show()

    #show top20 salary mean Jobtitle
    def showAVGSalary_plot(self,H1Info):
        avgWagePerJob = H1Info.groupby(['JOB_TITLE']).mean()['PREVAILING_WAGE'].nlargest(20).plot(kind="barh",color=tableau20)
        plt.title('Salary for Top20 jobs')
        plt.show()

    #show the difference between fulltime job and part time job.
    def showFullvsPart_plot(self,H1Info):
        fullTime = H1Info.FULL_TIME_POSITION.value_counts().plot(kind = 'bar',color=[(0.2,0.8,0.2),(0.8,0.2,0.2)])
        plt.ylabel('H1B number')
        plt.xlabel('JOb Type')
        plt.legend(loc="upper right", prop={'size':8})
        plt.title('Full time job VS Part-time job')
        plt.show()

    # CASE_STATUS situation predict. using decision Tree
    def predictCASE_STATUS(self,H1Info,job_TitleName,WORKSITE,EMPLOYER_NAME,PREVAILING_WAGE,dense):

        #top10JobList = H1Info['JOB_TITLE'].value_counts().sort_values(ascending=False).head(20).index.values
        top10JobDB = H1Info[H1Info['JOB_TITLE']==job_TitleName]
        #remove 2016, use other years as training day
        top10JobDB1 = top10JobDB[top10JobDB['YEAR'] != 2016]
        sampler = np.random.permutation(len(top10JobDB1)/dense)
        top10JobDB1 = top10JobDB1.take(sampler)


        #-----------  data process  -----------
        xSet1 = top10JobDB1.loc[:,'WORKSITE']
        WORKSITE_List = xSet1.drop_duplicates().values
        X_input1 = []
        for i in xSet1:
            X_input1.append(WORKSITE_List.tolist().index(i))

        xSet2 = top10JobDB1.loc[:,'EMPLOYER_NAME']
        EMPLOYER_NAME_List = xSet2.drop_duplicates().values
        X_input2 = []
        for i in xSet2:
            X_input2.append(EMPLOYER_NAME_List.tolist().index(i))

        xSet3 = top10JobDB1.loc[:,'CASE_STATUS']
        CASE_STATUS_List = xSet3.drop_duplicates().values
        X_input3 = []
        for i in xSet3:
            X_input3.append(CASE_STATUS_List.tolist().index(i))

        #salary range
        xSet4 = top10JobDB1.loc[:,'PREVAILING_WAGE']
        #define salary range
        salaryRange = [20000.00,40000.00,60000.00,80000.00,100000.00,120000.00,140000.00,
                       160000.00,180000.00,200000.00,400000.00,500000.00,1000000.00,2000000.00,float("inf")]
        X_input4 = []
        count = 0
        for i in xSet4:
            # deal with na !
            if isnan(i):
                i = 0
            for j in range(0,len(salaryRange)):
                if i <= salaryRange[j]:
                    X_input4.append(j)
                    break
                elif i > salaryRange[j]:
                    continue


        # -----------  decision Tree ALG  -----------
        x_input = np.zeros([3,len(X_input4)])
        x_input[0] = X_input1
        x_input[1] = X_input2
        x_input[2] = X_input4
        x_matrix = x_input.transpose()

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(x_matrix, X_input3)
        #predict
        salary_level = 0
        for j in range(0,len(salaryRange)):
                if PREVAILING_WAGE <= salaryRange[j]:
                    salary_level = j
                    break
                elif i > salaryRange[j]:
                    continue

        if WORKSITE in WORKSITE_List.tolist():
            worksite = WORKSITE_List.tolist().index(WORKSITE)
        else:
            worksite = 0
        if EMPLOYER_NAME in EMPLOYER_NAME_List.tolist():
            employ = EMPLOYER_NAME_List.tolist().index(EMPLOYER_NAME)
        else:
            employ = 0


        tree_result = clf.predict([[worksite, employ,salary_level ]])

        return CASE_STATUS_List[tree_result]


    # test one specify job's accuracy in 2016 base on 2011-2015 data
    def testDECISION_ACURACY(self,H1Info,job_TitleName,dense=10):
        top10JobDB = H1Info[H1Info['JOB_TITLE']==job_TitleName]
        # 2016data
        top10JobDB1 = top10JobDB[top10JobDB['YEAR'] == 2016]
        num_of_row = len(top10JobDB1.index)

        accuracy_count = 0
        for i in range(0,num_of_row,dense):
            work_site = top10JobDB1[i:i+1]['WORKSITE'].values
            emp_name = top10JobDB1[i:i+1]['EMPLOYER_NAME'].values
            wage = top10JobDB1[i:i+1]['PREVAILING_WAGE'].values
            if self.predictCASE_STATUS(top10JobDB,job_TitleName,work_site,emp_name,wage,dense) == top10JobDB1[i:i+1]['CASE_STATUS'].values:
                accuracy_count +=1

        #print 'accuracy_count'
        #print accuracy_count
        if num_of_row == 0:

            return 'no job data'
        else:

            #print num_of_row
            return float(accuracy_count)/num_of_row*dense

    #get top10 popular job's accuracy in 2016 base on 2011-2015
    def testTOP10JOB_acuracy(self,H1Info):
        top10 = ['COMPUTER PROGRAMMER','SYSTEMS ANALYST','SOFTWARE DEVELOPER','BUSINESS ANALYST','COMPUTER SYSTEMS ANALYST',
                'TECHNOLOGY LEAD - US','SENIOR SOFTWARE ENGINEER','TECHNOLOGY ANALYST - US','ASSISTANT PROFESSOR','SENIOR CONSULTANT']



        dense = 10

        accuracy_rate = []
        for item in top10:
            accuracy_rate.append(self.testDECISION_ACURACY(H1Info,item,dense))


        result = dict(zip(top10,accuracy_rate))



        print result

        return result













if __name__ == "__main__":

    h1bSystem = H1bInfo()
    try:
        H1Info = h1bSystem.read_process_data('h1b_history.csv')
    except IOError:
        print "Error: cannot find the path,pleas put data and program in the same folder"
    else:
        print "data file read successfully"



    function = 1
    k_meanResult = h1bSystem.K_meanAnalyze(H1Info)
    salary_Result = h1bSystem.salaryAnalyze(H1Info,k_meanResult)
    while function != 0:

        print "-----------------------------------------"
        print "1. showCASE_STATUS\n2. showWORKSITE\n3. showSALARY_table\n4. showSALARY_plot\n5. showTOP6com_table\n6. showYearTrend_plot\n7. showJOBTITLE_plot\n8. showAVGSalary_plot\n9. showFullvsPart_plot\n10.predic_showCASE_STATUS\n11.DecisionTreeAcuracy\n12.top10Accuracy"

        function = input("please choose the function(input 0 to exit):\n")

        if function == 1:
            h1bSystem.showCASE_STATUS(H1Info)
        elif function == 2:
            dense = input("please choose the data dense(1000/2000):\n")
            h1bSystem.showWORKSITE(dense,k_meanResult)
        elif function == 3:
            h1bSystem.showSALARY_table(salary_Result[0],salary_Result[1],salary_Result[2],salary_Result[3],salary_Result[4])
        elif function == 4:
            h1bSystem.showSALARY_plot(salary_Result[3],salary_Result[2])
        elif function == 5:
            h1bSystem.showTOP6com_table(H1Info)
        elif function == 6:
            h1bSystem.showYearTrend_plot(H1Info)
        elif function == 7:
            h1bSystem.showJOBTITLE_plot(H1Info)
        elif function == 8:
            h1bSystem.showAVGSalary_plot(H1Info)
        elif function == 9:
            h1bSystem.showFullvsPart_plot(H1Info)
        elif function == 10:
            job_TitleName = raw_input('please input your jobTitle:\n')
            WORKSITE = raw_input('please input your WORKSITE:\n')
            EMPLOYER_NAME = raw_input('please input your EMPLOYER_NAME:\n')
            PREVAILING_WAGE = raw_input('please input your PREVAILING_WAGE(number):\n')
            result = h1bSystem.predictCASE_STATUS(H1Info,job_TitleName,WORKSITE,EMPLOYER_NAME,float(PREVAILING_WAGE))
            print '-------------'
            print 'your future prediction is:'
            print result
            print '-------------'
        elif function == 11:
            job_TitleName = raw_input('please input your jobTitle:\n')
            result = h1bSystem.testDECISION_ACURACY(H1Info,job_TitleName)
            print 'acuracy rate is :'
            print result
        elif function == 12:
            h1bSystem.testTOP10JOB_acuracy(H1Info)
