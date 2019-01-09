

---

---



The idea behind this blogpost is to teach myself some data analysis in Python and hopefully be useful to someone who reads this as well.

My intent in this post is to extract H1B filing data for Machine Learning Engineers and answer a few questions about these job postings.

As a part of the H1B visa application process, employers need to file an Labor Condition Application(LCA) that is made publicly available to everyone. One can download all LCA applications for every year for all H1B applications from [Foreign Labor Website](https://www.foreignlaborcert.doleta.gov/h-1b.cfm)

More than 100K applications have been filed in the past few years, and downloading the individual CSV files for each of these years (from 2013 to 2018) and filtering out the data was beyond the scope of my current knowledge. I'm hoping to work with those records once I've understood how to use SQL better.

I've decided to get the data from another website [H1BData](http://www.h1bdata.info) since one can search for records based on job titles and then scrape them using Python & Beautiful Soup and analyse the scraped data.

I worked on a Jupyter Notebook to get the data and conduct the analysis. The code can be found [here](https://github.com/sanjaykmenon/h1b-ml-analysis)

Some of the insights that I found were:

There has been a significant increase in the number of ML engineer related H1B applications from 2013-2018 with a breakout of the sector in 2016

![png](/images/output_27_0.png)

The mean salary for these positions saw an overall uptick from 2013-2018 with a significant jump from 2015 to 2016.


![png](/images/output_29_0.png)


The number of days between the application submission date and the proposed start date is reducing from 2013 to 2018.

![png](/images/output_38_0.png)

I also found that most applications were submitted in the month of March possibly since USCIS accepts new H1B applications usually towards the end of March / beginning of April. One could also infer from this that more new H1B applications are being filed than H1B transfer applications which are used in cases where the applicant already has an existing H1B visa and is changing employers.

This is a very preliminary exploratory analysis of this data. In the future, I hope to use the entire data (and not just ML engineer related jobs) to identify salary and location trends for H1B jobs. This could include a comparison of median salaries of the locations where these applications were filed with H1B salaries to actually see if employers are bringing workers on visas with possibly lower paing jobs. 

