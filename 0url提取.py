import csv


def write_csv(file, mark):
    csv_reader = csv.reader(open(file, encoding='utf-8'))
    text = ""
    a, b, c = 0, 0, 0
    counter = 0
    lst = []
    for row in csv_reader:
        if counter == 0:
            for i in range(len(row)):
                if row[i] == '标题链接':
                    a = i
                elif row[i] == '时间':
                    b = i
        else:
            lst.append([row[a], row[b]])
        counter += 1
    # print(text)
    # print(counter)
    # 1. 创建文件对象（指定文件名，模式，编码方式）a模式 为 下次写入在这次的下一行
    with open("data/all_news_urls.csv", "a", encoding="utf-8", newline="") as f:
        # 2. 基于文件对象构建 csv写入对象
        csv_writer = csv.writer(f)
        # 3. 构建列表头
        name = ['mark', 'url', 'date']
        csv_writer.writerow(name)
        # 4. 写入csv文件内容
        for i in lst:
            z = [mark, i[0], i[1]]
            csv_writer.writerow(z)
        print("写入数据成功")
        # 5. 关闭文件
        f.close()


if __name__ == '__main__':
    # {1:金融,2:汽车,3:食品,4:房产,5:科技,6:健康,8:军事}
    file_lst = ["data/1新华金融_新华网.csv",
                "data/2新华汽车_新华网.csv",
                "data/3新华食品_新华网.csv",
                "data/4新华房产_新华网.csv",
                "data/5新华科技_新华网.csv",
                "data/6新华健康_新华网.csv",
                ]
    for path in file_lst:
        mark = file_lst.index(path) + 1
        write_csv(path, mark)
