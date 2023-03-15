import os
import matplotlib.pyplot as plt


def plot_learning_curve(episodes, records, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, records, linestyle='-', color='r')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)

    plt.savefig(figure_file, format='svg')
    plt.show()


def create_directory(path: str, sub_dirs: list):
    if sub_dirs:
        for sub_dir in sub_dirs:
            if os.path.exists(path + sub_dir):
                print(path + sub_dir + 'is already exist!')
            else:
                os.makedirs(path + sub_dir, exist_ok=True)
                print(path + sub_dir + ' create successfully!')
    else:
        if os.path.exists(path):
            print(path + 'is already exist!')
        else:
            os.makedirs(path, exist_ok=True)
            print(path + 'create successfully')