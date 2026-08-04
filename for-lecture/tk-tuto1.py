#tkinterモジュールをtkという名前を付けてインポート
import tkinter as tk

#変数rootにtkウインドウをセットし、サイズを400x300に指定
root = tk.Tk()
root.geometry("400x300")

#変数item1にラベルをセットする。
#この時、親オブジェクトをroot(つまりtkウインドウ)に指定し、
#テキストをItem 1と指定し、背景色をピンクに指定する。
item1 = tk.Label(master=root, text="Item 1", bg="pink")

#変数item1(つまりラベル)を親ウインドウにパック(詰めこむ)。
#詰め込む方向は上(side="top"、隙間は埋めない(fill="none")、
#占有領域の拡張も要求しない(expand=False)
item1.pack(side="top", fill="none", expand=False)

#実は上記3設定はpackのデフォルト値なので、単にitem1.pack()と引数無しで実行しても同じことである。
#今回は説明の都合上、フルで記載した。

#準備できたのでウインドを表示させる。
root.mainloop()