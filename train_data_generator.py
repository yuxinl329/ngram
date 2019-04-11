import csv
import codecs

count = 0
LANGUAGE_CODES = []

with codecs.open('./data/labels.txt', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        LANGUAGE_CODES.append(line[:3])

with codecs.open('./data/train/sentences.csv', 'r', encoding='utf-8', errors='ignore') as csvfile:
    readCSV = csv.reader(csvfile, delimiter='\t')
    for row in readCSV:
        count += 1
        if row[1] in LANGUAGE_CODES:
            filename = './data/train/languages/' + row[1] + '.txt'
            with codecs.open(filename, 'a', encoding='utf-8', errors='ignore') as file:
                file.write(row[2] + '\n')

        num = float("%0.2f" % (count / 80000 * 100))
        if (count % 50000 == 0):
            print(str(num) + '%')