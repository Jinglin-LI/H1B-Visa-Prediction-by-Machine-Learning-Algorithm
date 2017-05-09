### Dataset available at https://www.kaggle.com/nsharan/h-1b-visa (see "Download(106 MB)")

### Dependence&Lib
numpy
scipy
matplotlib
sklearn
pandas

### Running method
python h1bSystem.py (ensure h1b_history.csv in the same level folder )

### Menu description
##### 1. showCASE_STATUS
-   def showCASE_STATUS(self,H1Info)

    >   Analyze the H1Bs by the status of their visa applications and show the plot

##### 2. showWORKSITE
-   def K_meanAnalyze(self,H1Info)
    >     K-Means Clustering to seperate the h1b location

-   def showWORKSITE(self,dense,H1LatLong)

    >   show the plot of the h1b WORKSITE after k-mean
    >   dense is the parameter to control distance between point, when gainning the info from the H1LatLong

##### 3. showSALARY_table
-   def showSALARY_table(self,salaryMin,salaryMax,salaryMean,salaryMedian,salaryStd)

    > show the salary detail after k-mean process

##### 4. showSALARY_plot
-   def showSALARY_plot(self,salaryMedian,salaryMean)

    > Plotting and comparison to the median US salary (2015)

##### 5. showTOP10com_table
-   def showTOP6com_table(self,H1Info)

    > show top 6 company who has apply to the h1b for employee

##### 6. showYearTrend_plot
-   def showYearTrend_plot(self,H1Info)

    > show the h1b number in everyear's change

##### 7. showJOBTITLE_plot
-   def showJOBTITLE_plot(self,H1Info)

    > show top20 popular Jobtitle and top10 Worksites for H1-B Visa holders

##### 8. showAVGSalary_plot
-   def showAVGSalary_plot(self,H1Info)

    > show top20 salary mean Jobtitle

##### 9.showFullvsPart_plot
-   def showFullvsPart_plot(self,H1Info)

    > show the difference between fulltime job and part time job.

##### 10.predic_showCASE_STATUS
-   def predictCASE_STATUS(self,H1Info,job_TitleName,WORKSITE,EMPLOYER_NAME,PREVAILING_WAGE,dense)

    > CASE_STATUS situation predict. using decision Tree algorithm from sklearn

##### 11.DecisionTreeAcuracy\n12.top10Accuracy
-   def testDECISION_ACURACY(self,H1Info,job_TitleName,dense=10)

    > test one specify job's accuracy in 2016 base on 2011-2015 data

##### 12.top10Accuracy
-   def testDECISION_ACURACY(self,H1Info,job_TitleName,dense=10)
-   def testTOP10JOB_acuracy(self,H1Info)

    > get top10 popular job's accuracy in 2016 base on 2011-2015

