#Tkinter GUI　gridの説明

import tkinter as tk

class Application(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)

        self.master.title("ウィジェットの配置(grid)")     # ウィンドウタイトル
        self.master.geometry("300x180")       # ウィンドウサイズ(幅x高さ)

        #--------------------------------------------------------
        # ラベルの作成
        label1 = tk.Label(self.master, text = "ラベル1", bg = 'cyan1')
        label2 = tk.Label(self.master, text = "ラベル2", bg = 'green1')
        label3 = tk.Label(self.master, text = "ラベル3", bg = 'yellow1')
        label4 = tk.Label(self.master, text = "ラベル4", bg = 'pink1')
        label5 = tk.Label(self.master, text = "ラベル5", bg = 'MediumPurple1')
        label6 = tk.Label(self.master, text = "***ラベル6***", bg = 'LightSteelBlue1')

        #--------------------------------------------------------
        # gridでウィジェットの配置
        label1.grid(row = 0, column = 1, columnspan = 3, sticky = tk.W+tk.E)
        label2.grid(row = 0, column = 0, rowspan = 5, sticky = tk.N+tk.S)
        label3.grid(row = 1, column = 1)
        label4.grid(row = 1, column = 3)
        label5.grid(row = 2, column = 2)
        label6.grid(row = 3, column = 1, columnspan = 3)
        
        #引数の説明
        #column ウィジェットを配置する列番号（0始まり）を指定します
        #columnspan グリッドを横方向に結合する数を指定します
        #ipadx ウィジェットの内側の横方向の隙間を設定します
        #ipady ウィジェットの内側の縦方向の隙間を設定します
        #padx ウィジェットの外側の横方向の隙間を設定します
        #pady ウィジェットの外側の縦方向の隙間を設定します
        #row ウィジェットを配置する行番号（0始まり）を指定します
        #rowspan グリッドを縦方向に結合する数を指定します
        #sticky グリッド内のウィジェットを配置する位置
        #アンカーの機能にも似ていますが、例えば上下（tk.N+ tk.S）を指定すると、ウィジェットが上下方向にグリッド内いっぱいに広がります。【設定値】tk.N, tk.S, tk.W, tk.E, tk.NW, tk.NE, tk.SW, tk.SE, tk.NSEW
        #および上記組み合わせ（tk.N+ tk.Sなど）
        #--------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master = root)
    app.mainloop()