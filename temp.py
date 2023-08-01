import matplotlib.pyplot as plt
'''
train_acc_lst = [99.93135011441647, 99.94001199760048, 99.94900560938297, 99.9589771639546, 99.96873371547682,
                 99.97697797559665, 99.97768604261967, 99.9835593916975, 99.98725018327862, 99.9891836348395]
val_acc_lst = [43.47826086956522, 42.028985507246375, 41.54589371980676, 39.371980676328505, 37.43961352657005,
               35.26570048309179, 32.367149758454104, 28.019323671497588, 24.879227053140095, 23.67149758454106]

# the below is the latest one
# [43.47826086956522, 42.99516908212561, 42.99516908212561, 43.23671497584541, 42.99516908212561, 42.99516908212561, 42.99516908212561, 42.99516908212561, 42.99516908212561, 42.7536231884058]

restricted_samples_lst = [43.47826086956522, 43.71980676328502, 43.47826086956522, 43.71980676328502,
                          43.71980676328502, 44.44444444444444, 44.20289855072464, 44.20289855072464, 44.44444444444444, 44.20289855072464]

# train_acc_lst = [0.965675057208238, 0.5405034324942791, 0.554233409610984, 0.548741418764302, 0.585812356979405, 0.6077803203661327, 0.6354691075514874, 0.7135011441647597, 0.8491990846681923, 0.9610983981693364,0.994508009153318, 0.9961098398169337]
# val_acc_lst = [0.3888888888888889, 0.357487922705314, 0.35507246376811596, 0.36231884057971014, 0.37681159420289856, 0.37922705314009664, 0.35990338164251207, 0.3526570048309179, 0.40096618357487923, 0.4178743961352657,0.4251207729468599, 0.4251207729468599]

pca_vals = [5, 10, 20, 50, 75, 100, 200, 400, 800, 1600, 3200, 4000]

y1 = train_acc_lst
x1 = [item for item in range(0, len(y1))]

# plotting the line 1 points
plt.plot(x1, y1, label="Train Accuracy")

# line 2 points
y2 = val_acc_lst
x2 = [item for item in range(0, len(y2))]


# plotting the line 2 points
plt.plot(x2, y2, label="Dev Accuracy")

y3 = restricted_samples_lst
x3 = [item for item in range(0, len(y3))]


plt.plot(x3, y3, label="Modified Dev Accuracy")
# naming the x axis
plt.xlabel('Epochs')
# naming the y axis
plt.ylabel('Accuracy')
# giving a title to my graph
plt.title('Accuracy Vs Epochs')

# show a legend on the plot
plt.legend()
plt.grid()
# function to show the plot
plt.show()
'''

x = [0.00105625218055744, 0.001305879961491351, 0.12165751284190827,
     0.44398884863449317, 1.113690276283877, 2.0540303131820563, 3.972631560952487, 9.521822995017493, 11.79862472677026, 16.352646947094538, 19.425954630086416, 24.668786554492403, 62.14352401426151]
y = [0.305, 0.305, 0.379, 0.403, 0.423, 0.43,
     0.432, 0.431, 0.43, 0.45, 0.43, 0.43, 0.436]

for i in range(len(y)):
    y[i] = 100*y[i]
plt.plot(x, y)
# naming the x axis
plt.xlabel('Regularizer Constant (C)')
# naming the y axis
plt.ylabel('Accuracy')
# giving a title to my graph
plt.title('Accuracy Vs C')

# show a legend on the plot
plt.legend()
plt.grid()
# function to show the plot
plt.show()

'''
freq = {}
    for i in range(len(speech.train_labels)):
        if (speech.train_labels[i] in freq):
            freq[speech.train_labels[i]] += 1
        else:
            freq[speech.train_labels[i]] = 1

    for key in freq:
        print(f"The key is {key} and the number of samples is {freq[key]}")




The key is OBAMA_PRIMARY2008 and the number of samples is 771
The key is PAUL_PRIMARY2012 and the number of samples is 58
The key is MCCAIN_PRIMARY2008 and the number of samples is 370
The key is CLINTON_PRIMARY2008 and the number of samples is 1333
The key is ROMNEY_PRIMARY2008 and the number of samples is 64
The key is BACHMANN_PRIMARY2012 and the number of samples is 42
The key is GINGRICH_PRIMARY2012 and the number of samples is 166
The key is RICHARDSON_PRIMARY2008 and the number of samples is 309
The key is EDWARDS_PRIMARY2008 and the number of samples is 309
The key is GIULIANI_PRIMARY2008 and the number of samples is 219
The key is THOMPSON_PRIMARY2008 and the number of samples is 134
The key is HUCKABEE_PRIMARY2008 and the number of samples is 120
The key is ROMNEY_PRIMARY2012 and the number of samples is 154
The key is SANTORUM_PRIMARY2012 and the number of samples is 136
The key is PERRY_PRIMARY2012 and the number of samples is 56
The key is PAWLENTY_PRIMARY2012 and the number of samples is 38
The key is HUNTSMAN_PRIMARY2012 and the number of samples is 26
The key is BIDEN_PRIMARY2008 and the number of samples is 51
The key is CAIN_PRIMARY2012 and the number of samples is 14
'''
