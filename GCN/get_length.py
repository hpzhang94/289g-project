dataset = 'R8'
min_len = 10000
avg_len = 0
max_len = 0 
length = []

f = open('data/corpus/' + dataset + '_shuffle.txt', 'r')
lines = f.readlines()

for line in lines:
    temp = line.strip().split()
    length.append(str(len(temp)))
    avg_len = avg_len + len(temp)
    if len(temp) < min_len:
        min_len = len(temp)
    if len(temp) > max_len:
        max_len = len(temp)
f.close()
avg_len = 1.0 * avg_len / len(lines)

print('min_len : ' + str(min_len))
print('max_len : ' + str(max_len))
print('average_len : ' + str(avg_len))

f = open('./data/' + dataset + '.len.txt', 'w')
f.writelines('\n'.join(length))
f.close()