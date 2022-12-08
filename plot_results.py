import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def create_plot(filename):
    length = count_lines_enumrate(filename)
    x = np.zeros(length + 1)
    y = np.zeros((length + 1, 4))

    # put the csvs into arrays to plot 
    with open(filename) as data:
        csv_reader = csv.reader(data, delimiter=',')
        i = 0
        for row in csv_reader:
            if i != 0:
                x[i] = row[0]      # epoch
                y[i] = row[1:5]    # epsilon, steps, reward, loss
            i += 1

    # create the plots
    fig, axs = plt.subplots(2)
    fig.suptitle(f'{filename}')

    axs[0].set_title('Steps Taken')
    axs[0].plot(x, y[:,1], 'tab:red')

    axs[1].set_title('Reward')
    axs[1].plot(x, y[:,2], 'tab:orange')

    fig.subplots_adjust(hspace=0.5)

    # return the plot
    return fig


def count_lines_enumrate(file_name):
    # get numbner of lines in a csv file
    fp = open(file_name,'r')
    line_count = list(enumerate(fp))[-1][0]
    return line_count


def main():
    # loop through csv files in the current directory and save them as png
    for f in os.listdir('./'):
        if f.endswith('.csv'):
            fig = create_plot(f)
            fig.savefig(f'{f}.png')

main()