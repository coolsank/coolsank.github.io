

---

---



The idea behind this blogpost is to teach myself some data analysis in Python and hopefully be useful to someone who reads this as well.

My intent in this post is to extract H1B filing data for Machine Learning Engineers and answer a few questions about them.

As a part of the H1B visa application process, employers need to file an Labor Condition Application(LCA) that is made publicly available to everyone. One can download all LCA applications for every year for all H1B applications from [Foreign Labor Website](https://www.foreignlaborcert.doleta.gov/h-1b.cfm)






```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
```


```python
url='https://h1bdata.info/index.php?em=&job=machine+learning%2B&city=&year=All+Years'
```


```python
page = requests.get(url)
```



