import csv
...

with open("GFG.csv") as in_file:
    with open("newsHeadline.csv", 'w') as out_file:
        writer = csv.writer(out_file)
        for row in csv.reader(in_file):
            if row:
                writer.writerow(row)