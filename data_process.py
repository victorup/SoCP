import csv
import time


dev_emotion_file = '../../old_code/story_generation/data/storycommonsense_data/csv_version/dev/emotion/allcharlinepairs.csv'  # namely 'plutchik'
dev_motivation_file = '../../old_code/story_generation/data/storycommonsense_data/csv_version/dev/motiv/allcharlinepairs.csv'  # namely 'reiss'
test_emotion_file = '../../old_code/story_generation/data/storycommonsense_data/csv_version/test/emotion/allcharlinepairs.csv'  # namely 'plutchik'
test_motivation_file = '../../old_code/story_generation/data/storycommonsense_data/csv_version/test/motiv/allcharlinepairs.csv'  # namely 'reiss'

title = ['storyid', 'sentence', 'context', 'char', 'char_plutchik', 'char_maslow', 'char_reiss']
emotion_p = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
reiss = ['status', 'approval', 'tranquility', 'competition', 'health', 'family',
         'romance', 'food', 'indep', 'power', 'order', 'curiosity', 'serenity',
         'honor', 'belonging', 'contact', 'savings', 'idealism', 'rest']  # 19个reiss
maslow = ['physiological', 'stability', 'love', 'esteem', 'spiritual growth']


def read_emotion_file(filepath):
    # 总列表
    storyid = []
    sentence = []
    context = []
    char = []  # char总列表
    char_emotion = []
    char_reiss = []
    char_maslow = []

    l = 0  # linenum
    sid = ''  # 一句的id
    sen = ''  # 某一个sentence
    con = ''  # 某一个context
    ch = ''  # 一句中的一个char
    c = []  # 一句中的所有char
    char_e = {}  # 一句中某个char和其emotion的字典
    char_r = {}  # 一句中某个char和其reiss的字典
    char_m = {}
    c_e = []  # 包装一句中的所有char和emotion列表
    c_r = []  # 包装一句中的所有char和reiss列表
    c_m = []
    # count_out = []  # emotion得分
    # distribution = [0] * len(emotion_p)
    count_distribution = [0] * len(emotion_p)
    reiss_distribution = [0] * len(reiss)
    maslow_distribution = [0] * len(maslow)
    aff = ''  # affected(yes/no)

    with open(filepath, newline='') as csvfile:  # read the file
        data = csv.reader(csvfile)
        for idx, line in enumerate(data):
            '''不断更新：storyid, count_story'''
            # 第一个if记录storyid
            # 处理文件第一行title
            if idx == 0:  # title略过
                continue
            if idx == 1:
                sid = line[0]
                sen = line[5]
                con = line[4]
                ch = line[2]
                aff = line[6]  # affected(yes/no)
            # 处理最后一列的格式
            line[-1] = line[-1].replace("[", "")
            line[-1] = line[-1].replace("]", "")
            line[-1] = line[-1].replace('"', "")
            line[-1] = line[-1].split(",")  # 列表
            line[-2] = line[-2].replace("[", "")
            line[-2] = line[-2].replace("]", "")
            line[-2] = line[-2].replace('"', "")
            line[-2] = line[-2].split(",")  # 列表
            # 如果linenum不变
            if l == line[1]:
                sid = sid  # storyid不变
                sen = sen  # sentence不变
                con = con  # context不变
                # aff = aff  # ######??????????

                # 如果char不变
                if ch == line[2]:  # emotion得分添加
                    if filepath.split('/')[-2] == 'emotion':
                        for i in range(len(line[-1])):  # 遍历plutchik列表
                            if line[-1][i].strip().split(':')[0] in emotion_p:  # 如果plutchik已经存在emotion_p列表中
                                pos = emotion_p.index(line[-1][i].strip().split(':')[0])  # 找到第一次出现的索引
                                # distribution[pos] = distribution[pos] + 1  # 在plutchik出现次数分布上加1
                                count_distribution[pos] = count_distribution[pos] + int(
                                    line[-1][i].strip().split(':')[-1])  # plutchik的得分加1
                    else:
                        for i in range(len(line[-1])):
                            if line[-1][i] in reiss:
                                pos = reiss.index(line[-1][i])
                                reiss_distribution[pos] = reiss_distribution[pos] + 1
                        for i in range(len(line[-2])):
                            if line[-2][i] in maslow:
                                pos = maslow.index(line[-2][i])
                                maslow_distribution[pos] = maslow_distribution[pos] + 1


                # 如果char变了
                else:
                    sid = sid  # storyid不变
                    sen = sen  # sentence不变
                    con = con  # context不变
                    c.append(ch)  # 添加该句中的前一个char

                    if aff == 'no':  # 判断上一个affected
                        count_distribution = [0] * len(emotion_p)
                        char_e[ch] = count_distribution[:]  # 构成上一个char的emotion字典

                        reiss_distribution = [0] * len(reiss)
                        char_r[ch] = reiss_distribution[:]  # 切片是为了防止改变，修改了列表id

                        maslow_distribution = [0] * len(maslow)
                        char_m[ch] = maslow_distribution[:]

                    else:
                        char_e[ch] = count_distribution[:]  # 构成上一个char的emotion字典
                        count_distribution = [0] * len(emotion_p)

                        char_r[ch] = reiss_distribution[:]
                        reiss_distribution = [0] * len(reiss)

                        char_m[ch] = maslow_distribution[:]
                        maslow_distribution = [0] * len(maslow)
                    c_e.append(char_e)  # 将char_e字典添加到列表
                    c_r.append(char_r)
                    c_m.append(char_m)
                    aff = line[6]
                    ch = line[2]  # 更新char
                    char_e = {}  # 初始化字典
                    char_r = {}
                    char_m = {}

                    if filepath.split('/')[-2] == 'emotion':
                        for i in range(len(line[-1])):  # 遍历plutchik列表
                            if line[-1][i].strip().split(':')[0] in emotion_p:  # 如果plutchik已经存在emotion_p列表中
                                pos = emotion_p.index(line[-1][i].strip().split(':')[0])  # 找到第一次出现的索引
                                # distribution[pos] = distribution[pos] + 1  # 在plutchik出现次数分布上加1
                                count_distribution[pos] = count_distribution[pos] + int(
                                    line[-1][i].strip().split(':')[-1])  # plutchik的得分加1
                    else:
                        for i in range(len(line[-1])):
                            if line[-1][i] in reiss:
                                pos = reiss.index(line[-1][i])
                                reiss_distribution[pos] = reiss_distribution[pos] + 1
                        for i in range(len(line[-2])):
                            if line[-2][i] in maslow:
                                pos = maslow.index(line[-2][i])
                                maslow_distribution[pos] = maslow_distribution[pos] + 1

            # 如果linenum变了
            else:
                # if aff == 'no':
                #     count_distribution = [0] * len(emotion_p)
                storyid.append(sid + '_sen' + str(l))  # 添加前一个storyid
                sentence.append(sen)  # 添加前一个sentence
                context.append(con)  # 添加前一个context

                c.append(ch)  # 添加前一句的char
                char.append(c)  # 更新到总列表char。 列表套列表

                # 换行后，如果上一行的aff是no
                if aff == 'no':  # 判断上一个affected
                    count_distribution = [0] * len(emotion_p)
                    char_e[ch] = count_distribution[:]  # 构成上一个char的emotion字典

                    reiss_distribution = [0] *len(reiss)
                    char_r[ch] = reiss_distribution[:]

                    maslow_distribution = [0] * len(maslow)
                    char_m[ch] = maslow_distribution[:]
                    c_e.append(char_e)  # 将char_e字典添加到列表
                    c_r.append(char_r)
                    c_m.append(char_m)
                    aff = line[6]
                    ch = line[2]  # 更新char
                    char_e = {}  # 初始化字典
                    char_r = {}
                    char_m = {}
                # 换行后，如果上一行的aff是yes
                elif aff == 'yes':
                    char_e[ch] = count_distribution[:]  # 构成上一个char的emotion字典
                    c_e.append(char_e)  # 将char_e字典添加到列表
                    count_distribution = [0] * len(emotion_p)  # 换行后，先初始化，后面再计算

                    char_r[ch] = reiss_distribution[:]
                    c_r.append(char_r)
                    reiss_distribution = [0] * len(reiss)

                    char_m[ch] = maslow_distribution[:]
                    c_m.append(char_m)
                    maslow_distribution = [0] * len(maslow)

                    aff = line[6]
                    ch = line[2]  # 更新char
                    char_e = {}  # 初始化字典
                    char_r = {}
                    char_m = {}
                # 换行后，如果上一行的aff是其他
                else:
                    continue

                char_emotion.append(c_e)  # 添加到总列表
                char_reiss.append(c_r)
                char_maslow.append(c_m)

                # 更新新一行的数据
                l = line[1]  # 将linenum更新
                sid = line[0]
                sen = line[5]
                con = line[4]
                ch = line[2]
                c = []  # 清空char一句中的列表
                char_e = {}  # 清空一句中其中一个char的emotion字典
                char_r = {}
                char_m = {}
                c_e = []  # 清空一句的char_emotion列表
                c_r = []
                c_m = []
                aff = line[6]
                if filepath.split('/')[-2] == 'emotion':
                    for i in range(len(line[-1])):  # 遍历plutchik列表
                        if line[-1][i].strip().split(':')[0] in emotion_p:  # 如果plutchik已经存在emotion_p列表中
                            pos = emotion_p.index(line[-1][i].strip().split(':')[0])  # 找到第一次出现的索引
                            # distribution[pos] = distribution[pos] + 1  # 在plutchik出现次数分布上加1
                            count_distribution[pos] = count_distribution[pos] + int(
                                line[-1][i].strip().split(':')[-1])  # plutchik的得分加1
                else:
                    for i in range(len(line[-1])):
                        # char_reiss.append(str(line[-1][i]))
                        if line[-1][i] in reiss:
                            pos = reiss.index(line[-1][i])
                            reiss_distribution[pos] = reiss_distribution[pos] + 1
                    for i in range(len(line[-2])):
                        if line[-2][i] in maslow:
                            pos = maslow.index(line[-2][i])
                            maslow_distribution[pos] = maslow_distribution[pos] + 1

        # 直到将csv文件中每一行遍历完后：
        else:
            storyid.append(sid + '_sen' + str(l))  # 添加前一个storyid
            sentence.append(sen)  # 添加前一个sentence
            context.append(con)  # 添加前一个context

            c.append(ch)  # 添加前一句的char
            char.append(c)  # 更新到总列表char。 列表套列表

            char_e[ch] = count_distribution  # 将上一个char和其emotion构成字典
            char_r[ch] = reiss_distribution
            char_m[ch] = maslow_distribution
            c_e.append(char_e)  # 将字典嵌入列表中，表示一整句
            c_r.append(char_r)
            c_m.append(char_m)
            char_emotion.append(c_e)  # 添加到总列表
            char_reiss.append(c_r)
            char_maslow.append(c_m)
    return storyid, sentence, context, char, char_emotion, char_maslow, char_reiss


def main():
    time_start = time.time()

    write_row = []
    reiss_row = []
    maslow_row = []
    filepath = [dev_emotion_file, dev_motivation_file, test_emotion_file, test_motivation_file]
    for file in filepath:
        storyid, sentence, context, char, char_emotion, char_maslow, char_reiss = read_emotion_file(file)

        if file.split('/')[-2] == 'emotion':
            for i in range(len(storyid)):  # 遍历plutchik分布列表长度
                if i == 0:
                    continue
                for d in char_emotion[i]:
                    count = list(d.values())[0]
                    if sum(count) != 0:
                        for n, a in enumerate(count):  # 遍历列表，如果分数有小于4分的，那就将他们置为0
                            if a < 4:
                                count[n] = 0
                data_row = []
                data_row.append(storyid[i])
                data_row.append(sentence[i].replace('|', ' '))
                data_row.append(context[i])
                data_row.append(char[i])
                data_row.append(char_emotion[i])
                # data_row.append(char_reiss[i])
                write_row.append(data_row)
        else:
            for i in range(len(storyid)):  # 遍历plutchik分布列表长度
                if i == 0:
                    continue
                for d in char_reiss[i]:
                    count = list(d.values())[0]
                    if sum(count) != 0:
                        if 2 in count or 3 in count:  # 如果有2或3这种高分存在，那就将1分置为0，2和3都置为1
                            for n, a in enumerate(count):
                                if a == 1:
                                    count[n] = 0
                                if a == 2 or a == 3:
                                    count[n] = 1
                reiss_row.append(char_reiss[i])
                for d in char_maslow[i]:
                    count = list(d.values())[0]
                    if sum(count) != 0:
                        if 2 in count or 3 in count:
                            for n, a in enumerate(count):
                                if a == 1:
                                    count[n] = 0
                                if a == 2 or a == 3:
                                    count[n] = 1
                maslow_row.append(char_maslow[i])

    for j in range(len(write_row)):
        write_row[j].append(maslow_row[j])
        write_row[j].append(reiss_row[j])

    pro_data = 'data/pro_data.csv'
    with open(pro_data, 'w', newline='') as csvfile:
        data = csv.writer(csvfile)
        data.writerow(title)
        for line in write_row:
            data.writerow(line)

    time_end = time.time()
    total_time = time_end - time_start
    print('total time: ', total_time)


main()
