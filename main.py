import pandas
from bs4 import BeautifulSoup
df = pandas.read_csv("Resume.csv")
#only for it
df = df.loc[df["Category"]=="INFORMATION-TECHNOLOGY"]
print(df)
for row in df.values:
    data = row[2]
    parsed_data = BeautifulSoup(data)
    print(parsed_data.find('div').text)
