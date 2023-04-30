# p = [3, 1, 2, 0] # 4
# p = [5,2,1,3,6,8,4,7,9,0]
p = [3,2,1,5,4,6,8,7,9,10,13,15,12,16,17,19,18,0,11,14] # 20
p = [3,23,2,28,1,20, 5,4,6,21, 8,7,9,27, 10,13,24,15,26, 22,12,16,25,29, 17,19,18,0,11,14]
X = rd.randint(-100, 100, (len(p), len(p)))
times = 1000000  # 测试次数
# print(X)

start = time.perf_counter()
for i in range(times):
    Y = plain_permutation_function(X, p)
    # print(Y)
finish = time.perf_counter()
print((finish - start)/10000)

start = time.perf_counter()
for i in range(times):
    Y = permutation_function(X, p)
    # print(Y)
finish = time.perf_counter()
print((finish - start)/10000)

start = time.perf_counter()
for i in range(times):
    Y = permutation_function_two(X, p)
    # print(Y)
finish = time.perf_counter()
print((finish - start)/10000)

m = 30
d = 20
times = 1000
X = rd.randint(-100, 100, (m, d))

start = time.perf_counter()
for i in range(times):
    Y = plain_distance_function(X)
    #print(Y)
finish = time.perf_counter()
print((finish - start)/10000)

start = time.perf_counter()
for i in range(times):
    Y = distance_function(X)
    #print(Y)
finish = time.perf_counter()
print((finish - start)/10000)

start = time.perf_counter()
for i in range(times):
    Y = distance_function_two(X)
    #print(Y)
finish = time.perf_counter()
print((finish - start)/10000)