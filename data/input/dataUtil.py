# coding = utf-8
# -*- coding:utf-8 -*-
import json
import os
import config


def getEncoding(path):
    try:
        with open(path, 'r', encoding='utf-8') as fs:
            fs.readline()
            return 'utf-8'
    except UnicodeDecodeError:
        try:
            with open(path, 'r', encoding='ANSI') as fs:
                fs.readline()
                return 'ANSI'
        except UnicodeDecodeError:
            exit(-1)


def run():
    train_data = list()
    test_data = list()
    labels = dict()
    with open(config.train_with_label_path, 'r', encoding='utf-8') as fs:
        for line in fs:
            line = line.strip()
            if line[0] != 'g':
                idx = line.find(',')
                labels[int(line[0: idx])] = line[idx + 1:]

    with open(config.test_without_label_path, 'r', encoding='utf-8') as fs:
        for line in fs:
            line = line.strip()
            if line[0] != 'g':
                idx = line.find(',')
                labels[int(line[0: idx])] = ''

    for root, _, files in os.walk(config.raw_data_path):
        for f in files:
            if f[-1] == 't':
                # print(f)
                path = os.path.join(root, f)
                encoding = getEncoding(path)
                with open(path, 'r', encoding=encoding) as fs:
                    text = fs.read()
                    guid = int(f[0: f.find('.')])
                    # print(guid, encoding)
                    # print(text)

                    tag = labels.get(guid)
                    data = {
                        'guid': guid,
                        'text': text.strip(),
                        'tag': tag,
                        'img': str(guid) + '.jpg'
                    }
                    if tag is not None:
                        if tag != '':
                            train_data.append(data)
                        else:
                            test_data.append(data)
                    # print(text)

    print(len(train_data))
    print(len(test_data))

    with open(config.train_data_path, 'w', encoding='utf-8') as fs:
        json.dump(train_data, fs, ensure_ascii=False)
    with open(config.test_data_path, 'w', encoding='utf-8') as fs:
        json.dump(test_data, fs, ensure_ascii=False)


if __name__ == '__main__':
    run()
