

---

---



The idea behind this blogpost is to teach myself some data analysis in Python and hopefully be useful to someone who reads this as well.

My intent in this post is to extract H1B filing data for Machine Learning Engineers and answer a few questions about these job postings.

As a part of the H1B visa application process, employers need to file an Labor Condition Application(LCA) that is made publicly available to everyone. One can download all LCA applications for every year for all H1B applications from [Foreign Labor Website](https://www.foreignlaborcert.doleta.gov/h-1b.cfm)

More than 100K applications have been filed in the past few years, and downloading the individual CSV files for each of these years (from 2013 to 2018) and filtering out the data was beyond the scope of my current knowledge. I'm hoping to work with those records once I've understood how to use SQL better.

I've decided to get the data from another website [H1BData](http://www.h1bdata.info) since one can search for records based on job titles and then scrape them using Python & Beautiful Soup and analyse the scraped data.

I worked on a Jupyter Notebook to get the data and conduct the analysis. The code can be found here.



Initially, I imported the necessary libraries


```python
import requests 
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
```

I then added a variable url that had the [URL](https://h1bdata.info/index.php?em=&job=machine+learning%2B&city=&year=All+Years) link to the website that displayed the data I needed. This search query yielded all LCA applications related to machine learning position from the year 2013 to the year 2018.

```python
url='https://h1bdata.info/index.php?em=&job=machine+learning%2B&city=&year=All+Years'
```


```python
page = requests.get(url)
```


```python
soup = BeautifulSoup(page.text,'html.parser')
```


```python
table = soup.find_all(id='myTable')
```


```python
soup.prettify()
```



```python
table=table[0]
raw_data =[]
for row in table.find_all('tr'):
    for cell in row.find_all('td'):
        raw_data.append(cell.text)
```


```python
employer = []
np.array(employer.append(raw_data[::7]))
employer_df = pd.DataFrame(employer)
employer_df = employer_df.transpose()
```


```python
title = []
np.array(title.append(raw_data[1::7]))
title_df = pd.DataFrame(title)
title_df = title_df.transpose()
```


```python
salary = []
np.array(salary.append(raw_data[2::7]))
salary_df = pd.DataFrame(salary)
salary_df = salary_df.transpose()
salary_df = salary_df[0].str.replace(",", "").astype(float)
```


```python
location = []
np.array(location.append(raw_data[3::7]))
location_df = pd.DataFrame(location)
location_df = location_df.transpose()
```


```python
submit_date = []
np.array(submit_date.append(raw_data[4::7]))
submit_date_df = pd.DataFrame(submit_date)
submit_date_df = submit_date_df.transpose()
submit_date_df = pd.to_datetime(submit_date_df[0])
```


```python
start_date = []
np.array(start_date.append(raw_data[5::7]))
start_date_df = pd.DataFrame(start_date)
start_date_df = start_date_df.transpose()
start_date_df = pd.to_datetime(start_date_df[0])
```


```python
case_status = []
np.array(case_status.append(raw_data[6::7]))
case_status_df = pd.DataFrame(case_status)
case_status_df = case_status_df.transpose()
```


```python
df = pd.concat([employer_df,title_df,salary_df,location_df,submit_date_df,start_date_df,case_status_df], axis=1)

```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>0</th>
      <th>0</th>
      <th>0</th>
      <th>0</th>
      <th>0</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DROPBOX INC</td>
      <td>MACHINE LEARNING</td>
      <td>122429.0</td>
      <td>SAN FRANCISCO, CA</td>
      <td>2018-03-05</td>
      <td>2018-09-01</td>
      <td>CERTIFIED</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GRAVITY JACK INC</td>
      <td>MACHINE LEARNING &amp; COMPUTER VISION ENGINEER</td>
      <td>58500.0</td>
      <td>LIBERTY LAKE, WA</td>
      <td>2018-03-21</td>
      <td>2018-03-21</td>
      <td>DENIED</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GRAVITY JACK INC</td>
      <td>MACHINE LEARNING &amp; COMPUTER VISION ENGINEER</td>
      <td>58500.0</td>
      <td>LIBERTY LAKE, WA</td>
      <td>2018-03-26</td>
      <td>2018-03-26</td>
      <td>CERTIFIED</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JETLORE INC F/K/A QWHISPR INC</td>
      <td>MACHINE LEARNING &amp; DATA MINING ARCHITECT</td>
      <td>131000.0</td>
      <td>SAN MATEO, CA</td>
      <td>2017-03-15</td>
      <td>2017-09-14</td>
      <td>CERTIFIED</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MACHINE ZONE INC</td>
      <td>MACHINE LEARNING &amp; NLP DIRECTOR</td>
      <td>169749.0</td>
      <td>PALO ALTO, CA</td>
      <td>2017-03-20</td>
      <td>2017-09-16</td>
      <td>CERTIFIED</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns = ['company','position_title','salary','location','submit_date','start_date','case_status']
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>company</th>
      <th>position_title</th>
      <th>salary</th>
      <th>location</th>
      <th>submit_date</th>
      <th>start_date</th>
      <th>case_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DROPBOX INC</td>
      <td>MACHINE LEARNING</td>
      <td>122429.0</td>
      <td>SAN FRANCISCO, CA</td>
      <td>2018-03-05</td>
      <td>2018-09-01</td>
      <td>CERTIFIED</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GRAVITY JACK INC</td>
      <td>MACHINE LEARNING &amp; COMPUTER VISION ENGINEER</td>
      <td>58500.0</td>
      <td>LIBERTY LAKE, WA</td>
      <td>2018-03-21</td>
      <td>2018-03-21</td>
      <td>DENIED</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GRAVITY JACK INC</td>
      <td>MACHINE LEARNING &amp; COMPUTER VISION ENGINEER</td>
      <td>58500.0</td>
      <td>LIBERTY LAKE, WA</td>
      <td>2018-03-26</td>
      <td>2018-03-26</td>
      <td>CERTIFIED</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JETLORE INC F/K/A QWHISPR INC</td>
      <td>MACHINE LEARNING &amp; DATA MINING ARCHITECT</td>
      <td>131000.0</td>
      <td>SAN MATEO, CA</td>
      <td>2017-03-15</td>
      <td>2017-09-14</td>
      <td>CERTIFIED</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MACHINE ZONE INC</td>
      <td>MACHINE LEARNING &amp; NLP DIRECTOR</td>
      <td>169749.0</td>
      <td>PALO ALTO, CA</td>
      <td>2017-03-20</td>
      <td>2017-09-16</td>
      <td>CERTIFIED</td>
    </tr>
  </tbody>
</table>
</div>




```python
#considering only certified LCA applications. Other applications 
#are either withdrawn or denied.
df = df[df.case_status == 'CERTIFIED']
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>company</th>
      <th>position_title</th>
      <th>salary</th>
      <th>location</th>
      <th>submit_date</th>
      <th>start_date</th>
      <th>case_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DROPBOX INC</td>
      <td>MACHINE LEARNING</td>
      <td>122429.0</td>
      <td>SAN FRANCISCO, CA</td>
      <td>2018-03-05</td>
      <td>2018-09-01</td>
      <td>CERTIFIED</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GRAVITY JACK INC</td>
      <td>MACHINE LEARNING &amp; COMPUTER VISION ENGINEER</td>
      <td>58500.0</td>
      <td>LIBERTY LAKE, WA</td>
      <td>2018-03-26</td>
      <td>2018-03-26</td>
      <td>CERTIFIED</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JETLORE INC F/K/A QWHISPR INC</td>
      <td>MACHINE LEARNING &amp; DATA MINING ARCHITECT</td>
      <td>131000.0</td>
      <td>SAN MATEO, CA</td>
      <td>2017-03-15</td>
      <td>2017-09-14</td>
      <td>CERTIFIED</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MACHINE ZONE INC</td>
      <td>MACHINE LEARNING &amp; NLP DIRECTOR</td>
      <td>169749.0</td>
      <td>PALO ALTO, CA</td>
      <td>2017-03-20</td>
      <td>2017-09-16</td>
      <td>CERTIFIED</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ABNORMAL SECURITY INC</td>
      <td>MACHINE LEARNING &amp; USER EXPERIENCE ENGINEER</td>
      <td>165000.0</td>
      <td>SAN FRANCISCO, CA</td>
      <td>2018-04-09</td>
      <td>2018-05-21</td>
      <td>CERTIFIED</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Questions to be answered
#What is the mean income between 2013-2018?
#what is the comon salary (mode) for 2013-2018?
#how many ML jobs between these years?
#common time of application?
#trends between these years.
#additional things:
# remove all withdrawn petitions
#split city and state
#add column for median / average income by city if available
#total number of applications vs. certified applications
```


```python
df['submitdate_year'] = pd.DatetimeIndex(df['submit_date']).year
```


```python
df['submitdate_month'] = pd.DatetimeIndex(df['submit_date']).month
```


```python
df['startdate_year'] = pd.DatetimeIndex(df['start_date']).year
```


```python
df['startdate_month'] = pd.DatetimeIndex(df['start_date']).month
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>company</th>
      <th>position_title</th>
      <th>salary</th>
      <th>location</th>
      <th>submit_date</th>
      <th>start_date</th>
      <th>case_status</th>
      <th>submitdate_year</th>
      <th>submitdate_month</th>
      <th>startdate_year</th>
      <th>startdate_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DROPBOX INC</td>
      <td>MACHINE LEARNING</td>
      <td>122429.0</td>
      <td>SAN FRANCISCO, CA</td>
      <td>2018-03-05</td>
      <td>2018-09-01</td>
      <td>CERTIFIED</td>
      <td>2018</td>
      <td>3</td>
      <td>2018</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GRAVITY JACK INC</td>
      <td>MACHINE LEARNING &amp; COMPUTER VISION ENGINEER</td>
      <td>58500.0</td>
      <td>LIBERTY LAKE, WA</td>
      <td>2018-03-26</td>
      <td>2018-03-26</td>
      <td>CERTIFIED</td>
      <td>2018</td>
      <td>3</td>
      <td>2018</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JETLORE INC F/K/A QWHISPR INC</td>
      <td>MACHINE LEARNING &amp; DATA MINING ARCHITECT</td>
      <td>131000.0</td>
      <td>SAN MATEO, CA</td>
      <td>2017-03-15</td>
      <td>2017-09-14</td>
      <td>CERTIFIED</td>
      <td>2017</td>
      <td>3</td>
      <td>2017</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MACHINE ZONE INC</td>
      <td>MACHINE LEARNING &amp; NLP DIRECTOR</td>
      <td>169749.0</td>
      <td>PALO ALTO, CA</td>
      <td>2017-03-20</td>
      <td>2017-09-16</td>
      <td>CERTIFIED</td>
      <td>2017</td>
      <td>3</td>
      <td>2017</td>
      <td>9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ABNORMAL SECURITY INC</td>
      <td>MACHINE LEARNING &amp; USER EXPERIENCE ENGINEER</td>
      <td>165000.0</td>
      <td>SAN FRANCISCO, CA</td>
      <td>2018-04-09</td>
      <td>2018-05-21</td>
      <td>CERTIFIED</td>
      <td>2018</td>
      <td>4</td>
      <td>2018</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.submitdate_year.value_counts().sort_index()
```




    2013      8
    2014     31
    2015     42
    2016     42
    2017    125
    2018    242
    Name: submitdate_year, dtype: int64




```python
df.submitdate_year.value_counts().sort_index().plot()
plt.xlabel('Years')
plt.ylabel('Number of Applications')
plt.show()
```


![png](/images/output_27_0.png)



```python
ts = [2013,2014,2015,2016,2017,2018]

year2013 = df.loc[df.submitdate_year == ts[0],:]
salary_2013 = year2013.salary.mean()

year2014 = df.loc[df.submitdate_year == ts[1],:]
salary_2014 = year2014.salary.mean()

year2015 = df.loc[df.submitdate_year == ts[2],:]
salary_2015 = year2015.salary.mean()


year2016 = df.loc[df.submitdate_year == ts[3],:]
salary_2016 = year2016.salary.mean()

year2017 = df.loc[df.submitdate_year == ts[4],:]
salary_2017 = year2017.salary.mean()


year2018 = df.loc[df.submitdate_year == ts[5],:]
salary_2018 = year2018.salary.mean()

salary = [salary_2013,salary_2014,salary_2015,salary_2016,salary_2017,salary_2018]

```


```python
plt.plot(ts, salary)
plt.xlabel('Years')
plt.ylabel('Mean Salary  in USD')
plt.show()
```


![png](/images/output_29_0.png)



```python
print(time)
```


```python
df.shape
```




    (490, 11)




```python
average_days = []
i = 0
for i, row in df.iterrows():
    x = pd.Timedelta(df['start_date'][i] - df['submit_date'][i]).days
    average_days.append(x)
    i += 1

```


```python
len(average_days)
```




    490




```python
average_days = pd.DataFrame(average_days)
```


```python
df['days_between_submission_and_start'] = average_days
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>company</th>
      <th>position_title</th>
      <th>salary</th>
      <th>location</th>
      <th>submit_date</th>
      <th>start_date</th>
      <th>case_status</th>
      <th>submitdate_year</th>
      <th>submitdate_month</th>
      <th>startdate_year</th>
      <th>startdate_month</th>
      <th>days_between_submission_and_start</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DROPBOX INC</td>
      <td>MACHINE LEARNING</td>
      <td>122429.0</td>
      <td>SAN FRANCISCO, CA</td>
      <td>2018-03-05</td>
      <td>2018-09-01</td>
      <td>CERTIFIED</td>
      <td>2018</td>
      <td>3</td>
      <td>2018</td>
      <td>9</td>
      <td>180.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GRAVITY JACK INC</td>
      <td>MACHINE LEARNING &amp; COMPUTER VISION ENGINEER</td>
      <td>58500.0</td>
      <td>LIBERTY LAKE, WA</td>
      <td>2018-03-26</td>
      <td>2018-03-26</td>
      <td>CERTIFIED</td>
      <td>2018</td>
      <td>3</td>
      <td>2018</td>
      <td>3</td>
      <td>183.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JETLORE INC F/K/A QWHISPR INC</td>
      <td>MACHINE LEARNING &amp; DATA MINING ARCHITECT</td>
      <td>131000.0</td>
      <td>SAN MATEO, CA</td>
      <td>2017-03-15</td>
      <td>2017-09-14</td>
      <td>CERTIFIED</td>
      <td>2017</td>
      <td>3</td>
      <td>2017</td>
      <td>9</td>
      <td>180.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MACHINE ZONE INC</td>
      <td>MACHINE LEARNING &amp; NLP DIRECTOR</td>
      <td>169749.0</td>
      <td>PALO ALTO, CA</td>
      <td>2017-03-20</td>
      <td>2017-09-16</td>
      <td>CERTIFIED</td>
      <td>2017</td>
      <td>3</td>
      <td>2017</td>
      <td>9</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ABNORMAL SECURITY INC</td>
      <td>MACHINE LEARNING &amp; USER EXPERIENCE ENGINEER</td>
      <td>165000.0</td>
      <td>SAN FRANCISCO, CA</td>
      <td>2018-04-09</td>
      <td>2018-05-21</td>
      <td>CERTIFIED</td>
      <td>2018</td>
      <td>4</td>
      <td>2018</td>
      <td>5</td>
      <td>28.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
year2013 = df.loc[df.submitdate_year == ts[0],:]
year2014 = df.loc[df.submitdate_year == ts[1],:]
year2015 = df.loc[df.submitdate_year == ts[2],:]
year2016 = df.loc[df.submitdate_year == ts[3],:]
year2017 = df.loc[df.submitdate_year == ts[4],:]
year2018 = df.loc[df.submitdate_year == ts[5],:]


days_2013 = year2013.days_between_submission_and_start.mean()
days_2014 = year2014.days_between_submission_and_start.mean()
days_2015 = year2015.days_between_submission_and_start.mean()
days_2016 = year2016.days_between_submission_and_start.mean()
days_2017 = year2017.days_between_submission_and_start.mean()
days_2018 = year2018.days_between_submission_and_start.mean()

days_difference = [days_2013,days_2014,days_2015,days_2016,days_2017,days_2018]
```


```python
plt.plot(ts, days_difference)
plt.xlabel('Years')
plt.ylabel('Days between Application Submission & Job Start')
plt.show()
```


![png](/images/output_38_0.png)



```python
df.submitdate_month.mode()
```




    0    3
    dtype: int64




```python
df.startdate_month.mode()

```





