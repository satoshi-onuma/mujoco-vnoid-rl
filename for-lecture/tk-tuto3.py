#TkinterでのGUIレイアウトの詳細説明
#https://thom.hateblo.jp/entry/2022/02/08/000829

import tkinter as tk

root = tk.Tk()
root.geometry("400x300")

# フレームを準備
header = tk.Frame(master=root, bg="pink")
footer = tk.Frame(master=root, bg="lightblue")
container1 = tk.Frame(master=root, bg="lightgreen")
container2 = tk.Frame(master=root, bg="khaki")
container3 = tk.Frame(master=root, bg="mediumpurple1")

# フレームレイアウト
header.pack(side="top", fill="both", expand=False)
footer.pack(side="bottom", fill="both", expand=False)
container1.pack(side="left", fill="both", expand=True)
container2.pack(side="left", fill="both", expand=True)
container3.pack(side="left", fill="both", expand=True)

# ヘッダー内レイアウト
title_label = tk.Label(master=header, text = "Sample App", bg=header["bg"])
title_label.pack()

# フッター内レイアウト
creator_label = tk.Label(master=footer, text = "Created by thom.", bg=footer["bg"])
creator_label.pack()

root.mainloop()