#GUIレイアウト，placeの例
#https://thom.hateblo.jp/entry/2019/09/29/122444

import tkinter as tk
import winsound
from time import sleep

root_window = tk.Tk()
root_window.geometry("200x160")

def start_clicked():
    replay = int(ent_replay.get())
    count = int(ent_count.get())
    high_tone = int(ent_high_tone.get())
    low_tone = int(ent_low_tone.get())
    duration = int(ent_duration.get())
    for i in range(replay):
        for j in range(count):
            winsound.Beep(55*2**low_tone,duration)
            sleep(1)
        winsound.Beep(55*2**high_tone,duration*2)
        sleep(1)

lbl_replay = tk.Label(
    master=root_window,
    text="Replay")

lbl_count = tk.Label(
    master=root_window,
    text="Count")

lbl_high_tone = tk.Label(
    master=root_window,
    text="High tone")

lbl_low_tone = tk.Label(
    master=root_window,
    text="Low tone")

lbl_duration = tk.Label(
    master=root_window,
    text="Duration")

ent_replay = tk.Entry(
    master=root_window,
    width = 10)

ent_count = tk.Entry(
    master=root_window,
    width = 10)

ent_high_tone = tk.Entry(
    master=root_window,
    width = 10)

ent_low_tone = tk.Entry(
    master=root_window,
    width = 10)

ent_duration = tk.Entry(
    master=root_window,
    width = 10)

btn = tk.Button(
    master=root_window,
    text="Start",
    command=start_clicked
    )


lbl_replay.place(x=5,y=5)
ent_replay.place(x=70,y=5)

lbl_count.place(x=5,y=30)
ent_count.place(x=70,y=30)

lbl_high_tone.place(x=5,y=55)
ent_high_tone.place(x=70,y=55)

lbl_low_tone.place(x=5,y=80)
ent_low_tone.place(x=70,y=80)

lbl_duration.place(x=5,y=105)
ent_duration.place(x=70,y=105)

btn.place(x=70,y=130)

#プログラムで文字を入力するには、insert命令で、最初の引数 tk.END はテキストボックスの最後に挿入するという意味で、2番目の引数が実際に入力する値
ent_replay.insert(tk.END,"5")
ent_count.insert(tk.END,"3")
ent_high_tone.insert(tk.END,"5")
ent_low_tone.insert(tk.END,"3")
ent_duration.insert(tk.END,"200")

root_window.mainloop()