import numpy as np

from HW01.part1 import run


def save_file():
    fw = open('randn.txt', 'a')  # output file
    data1=run(1,4,2000)

    for data in data1:
        fw.write(str(data)+"\n")  # write to file
    #     fw.write(data1)  # write to file
    data2=run(4,9,1000)

    for data in data2:
        fw.write(str(data) + "\n")  # write to file



if __name__ == '__main__':
    save_file()

    data1 = run(1, 4, 2000)
    data2 = run(4, 9, 1000)
    data1= np.array(data1)
    data2= np.array(data2)



    data3=np.concatenate((data1,data2))


    print("mean is : ", np.mean(data3))
    print("var is : ",np.var(data3))

