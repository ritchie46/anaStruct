import csv

with open("hea.csv") as f:
    r = csv.reader(f)
    HEA = {}

    headers = next(r)
    next(r)  # units not needed
    for row in r:
        params = {}

        for i in range(1, len(row)):
            params[headers[i]] = float(row[i])

        HEA[int(row[0])] = params


