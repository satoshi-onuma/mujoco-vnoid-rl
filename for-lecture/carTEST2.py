import tkinter as tk

class Car:
    #コンストラクタ(この中にCarの属性を定義)
    def __init__(self, name, color, number):
        self.name = name
        self.color = color
        self.number = number
        self.nowSpeed = 0
        self.xPos = 0
        self.yPos = 0
        self.nowPassenger = 0
        self.remainTime = 10
    #アクセル
    def accel_speed_ByCar(self):
        self.nowSpeed = self.nowSpeed + 10
    #減速
    def deccel_speed_ByCar(self):
        self.nowSpeed = self.nowSpeed - 5
        

now_speed = 0
your_car = None
car1 = None
car2 = None
car3 = None
car4 = None
car5 = None

# car1〜5の代わりに1つのリストを用意（初期値は空）
other_cars = []

def register_name():
    global your_car
    global car1
    global car2
    global car3
    global car4
    global car5
    global other_cars  # リストを使用することを宣言
    user_input = entry_name.get()
    label.config(text=f"あなたの名前は{user_input}！ようこそ")
    #ユーザのCarを実体化（オブジェクトを生成）する（your_carの初期化）
    your_car = Car(f"{user_input}", "white", "1234")
    car1 = Car("taro", "white", "2345")
    car2 = Car("jiro", "black", "3456")
    car3 = Car("saburo", "orange", "4567")
    car4 = Car("hanako", "yellow", "6789")
    car5 = Car("mask", "red", "0000")
    
    # リストに5つのCarオブジェクトを格納
    other_cars = [
        Car("taro", "white", "2345"),     # インデックス0 (入力: 1)
        Car("jiro", "black", "3456"),     # インデックス1 (入力: 2)
        Car("saburo", "orange", "4567"),  # インデックス2 (入力: 3)
        Car("hanako", "yellow", "6789"),  # インデックス3 (入力: 4)
        Car("mask", "red", "0000")        # インデックス4 (入力: 5)
    ]

def info_car():
    global your_car  # 外側の変数を使用することを宣言
    global other_cars  # リストを使用することを宣言
    #label.config(text=f"車の情報：名前：{your_car.name}，色：{your_car.color}")
    label.config(text=f"車の情報：名前：{other_cars[3].name}，色：{other_cars[3].color}")
    
def info_car2():
    global car1
    global car2
    global car3
    global car4
    global car5
    user_input = entry_name2.get()
    if user_input == "1":
        label.config(text=f"車の情報：名前：{car1.name}，色：{car1.color}, 現在速度：{car1.nowSpeed}")
    elif user_input == "2":
        label.config(text=f"車の情報：名前：{car2.name}，色：{car2.color}, 現在速度：{car2.nowSpeed}")
    elif user_input == "3":
        label.config(text=f"車の情報：名前：{car3.name}，色：{car3.color}, 現在速度：{car3.nowSpeed}")
    elif user_input == "4":
        label.config(text=f"車の情報：名前：{car4.name}，色：{car4.color}, 現在速度：{car4.nowSpeed}")
    elif user_input == "5":
        label.config(text=f"車の情報：名前：{car5.name}，色：{car5.color}, 現在速度：{car5.nowSpeed}")

def accel_speed():
    #global now_speed  # 外側の変数を使用することを宣言
    #now_speed = now_speed + 10
    #label_NowSpeed.config(text=f"{now_speed}")
    global your_car
    #your_carのアクセルを実行する
    your_car.accel_speed_ByCar()
    label_NowSpeed.config(text=f"{your_car.nowSpeed}")

def deccel_speed():
    #global now_speed  # 外側の変数を使用することを宣言
    #now_speed = now_speed - 5
    #label_NowSpeed.config(text=f"{now_speed}")
    global your_car
    #your_carの減速を実行する
    your_car.deccel_speed_ByCar()
    label_NowSpeed.config(text=f"{your_car.nowSpeed}")

# ウィンドウの作成
root = tk.Tk()
root.title("Hello Car")
root.geometry("400x300")

# ラベルの作成
label = tk.Label(root, text="あなたの名前")
label.place(x=5, y=5)
# エントリーウィジェットの作成
entry_name = tk.Entry(root, width=20)
entry_name.place(x=5, y=40)
#登録ボタン
register_button = tk.Button(root, text="登録", command=register_name)
register_button.place(x=140,y=40)
#車の情報を表示するボタン
register_button = tk.Button(root, text="情報", command=info_car)
register_button.place(x=180,y=40)
# ラベルの作成
label2 = tk.Label(root, text="現在の速度")
label2.place(x=5,y=80)
label_NowSpeed = tk.Label(root, text="???")
label_NowSpeed.place(x=100,y=80)
#アクセルボタン
accel_button = tk.Button(root, text="アクセル", command=accel_speed)
accel_button.place(x=5,y=120)
#ブレーキボタン
deccel_button = tk.Button(root, text="ブレーキ", command=deccel_speed)
deccel_button.place(x=70,y=120)

# エントリーウィジェットの作成
entry_name2 = tk.Entry(root, width=20)
entry_name2.place(x=5, y=200)
#情報ボタン
register_button = tk.Button(root, text="情報", command=info_car2)
register_button.place(x=140,y=200)

# メインループの実行
root.mainloop()