import pickle
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import wfdb
import fiducial_features
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tkinter import filedialog
import pickle
import fiducial_features
import asyncio

import non_fiducial

fig = plt.Figure(figsize=(6, 4), dpi=100)
fig1 = plt.Figure(figsize=(6, 4), dpi=100)
window = tk.Tk()

def button_click1():
    file_path = tk.filedialog.askopenfilename()
    fig.clf()
    canvas = FigureCanvasTkAgg(figure=fig1, master=window)
    canvas.get_tk_widget().destroy()
    # Display the selected file path
    if file_path:
        patient = file_path.split('/')[6]
        bonus = file_path.split('/')[5]
        label = np.array([patient])
        data = np.load(file_path,allow_pickle=True)
        ax1 = fig1.add_subplot(111)
        ax1.plot(np.arange(0, len(data)), data)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Signal')
        canvas = FigureCanvasTkAgg(figure=fig1, master=window)
        canvas.draw()
        canvas.get_tk_widget().place(x=200, y=60)
        print(data.shape)
        if bonus=='new_non_fiducial_Bonus':
            data = data.reshape(1,6)
            result = test_bonus(data, label)
        else:
            data = data.reshape((1, 500))
            result = test_non(data, label)
        if result >= 0.7:
            l = tk.Label(window, text="The patient has been identified:" + patient, font=("Arial", 20), fg="blue",
                         bg="yellow")
            l.place(x=110, y=500)
        else:
            l = tk.Label(window, text="The patient has not been identified", font=("Arial", 20), fg="blue", bg="yellow")
            l.place(x=110, y=500)
def button_click():
    file_path = tk.filedialog.askopenfilename()
    fig.clf()
    canvas = FigureCanvasTkAgg(figure=fig,master=window)
    canvas.get_tk_widget().destroy()
    # Display the selected file path
    if file_path:
        patient = file_path.split('/')[6]
        label = np.array([patient])
        points,segment = fiducial_features.load_signal(file_path)
        data = np.array(fiducial_features.calculate_feature(segment,points))
        data = data.reshape((1,5))
        ax = fig.add_subplot(111)
        ax.plot(np.arange(0,len(segment)),segment)
        ax.scatter(points,segment[points],marker='o',color='red')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.set_title('Signal')
        canvas = FigureCanvasTkAgg(figure=fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().place(x=200,y=60)
        result = test(data, label)
        if result >= 0.9:
            l = tk.Label(window, text="The patient has been identified:"+patient, font=("Arial", 20), fg="blue", bg="yellow")
            l.place(x=110, y=500)
        else:
            l = tk.Label(window, text="The patient has not been identified", font=("Arial", 20), fg="blue", bg="yellow")
            l.place(x=110, y=500)


def create_window():
    window.geometry("800x600")
    button_fiducial = tk.Button(window, text="Open Signal fiducial")
    button_fiducial.bind("<Button-1>", lambda event: button_click())
    button_fiducial.place(x=20,y=20)
    button_non_fiducial = tk.Button(window, text="Open Signal non fiducial")
    button_non_fiducial.bind("<Button-1>", lambda event: button_click1())
    button_non_fiducial.place(x=20, y=60)
    button_bonus = tk.Button(window, text="Open Signal bonus")
    button_bonus.bind("<Button-1>", lambda event: button_click1())
    button_bonus.place(x=20, y=100)
    window.mainloop()
def train(labels,data,random_state):
    data = np.array(data)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=random_state,test_size=0.2)
    clf = SVC(kernel='linear', max_iter=50)
    clf.fit(x_train, y_train)
    filename = "model_fiducial.sav"
    pickle.dump(clf,open(filename,'wb'))
def test(data,labels):
    loaded_model = pickle.load(open('model_fiducial.sav', 'rb'))
    result = loaded_model.score(data, labels)
    return result
def test_non(data,labels):
    loaded_model = pickle.load(open('model_non_fiducial.sav', 'rb'))
    result = loaded_model.score(data, labels)
    return result
def test_bonus(data,labels):
    loaded_model = pickle.load(open('model_non_fiducial_Bonus.sav', 'rb'))
    result = loaded_model.score(data, labels)
    return result
def main():
    create_window()
main()