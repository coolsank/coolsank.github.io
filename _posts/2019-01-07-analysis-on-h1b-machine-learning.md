

---
title:  "Basic Exploratory Analysis on H1B Filings"
---



This is a test type up. I fthis works I continue.



```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
```
{% highlight python %}
x = ('a', 1, False)
{% endhighlight %}



```python
url='https://h1bdata.info/index.php?em=&job=machine+learning%2B&city=&year=All+Years'
```


```python
page = requests.get(url)
```



