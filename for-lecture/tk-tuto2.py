#ラベルを増やすには、単純にitem変数をもう一つ準備して、そちらもパックすれば良い。

import tkinter as tk

root = tk.Tk()
root.geometry("400x300")

item1 = tk.Label(master=root, text="Item 1", bg="pink")
item2 = tk.Label(master=root, text="Item 2", bg="lightblue")

item1.pack(side="top", fill="none", expand=False)
item2.pack(side="top", fill="none", expand=False)

root.mainloop()