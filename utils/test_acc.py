import pandas as pd

count = 0
correct_pd = pd.read_csv("./test_correct.txt", sep=" ", header=None, names=['ImageName', 'label'])
test_pd = pd.read_csv("./Resnet18-Trained-SGD-V1-data_balance_test.csv", sep=" ", header=None, names=['ImageName', 'label'])

for i in range(320):
    if correct_pd.label[i] == test_pd.label[i]:
        count = count + 1
    else:
        print("Error image: " + str(test_pd.ImageName[i]) +". Error label: " + str(test_pd.label[i]))
print("=====================================================")
print("Test accuracy is %.3f" %(count/320.0))
print("=====================================================")


