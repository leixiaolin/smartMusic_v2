import csv

sFileName='./all_singer.csv'

with open(sFileName,newline='',encoding='UTF-8') as csvfile:
    rows=csv.reader(csvfile)
    i = 0
    for row in rows:
        print(','.join(row))
        i += 1
    print(i)