import csv
import codecs

LANGUAGE_CODES = []

with codecs.open('./data/labels.txt', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        LANGUAGE_CODES.append(line[:3])

with codecs.open('./data/test/LanideNN_testset.txt', 'r', encoding='utf-8', errors='ignore') as file1:
    with codecs.open('./data/test/test_sentences.txt', 'w', encoding='utf-8', errors='ignore') as file2:
        with codecs.open('./data/test/test_labels.txt', 'w', encoding='utf-8', errors='ignore') as file3:
            for line in file1:
                if line[:3] in LANGUAGE_CODES:
                    file2.write(line[4:])
                    file3.write(line[:3] + '\n')
                elif line[:3] == 'zho':
                    file2.write(line[4:])
                    file3.write('cmn' + '\n')

