#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>   // PythonのNumpy配列を扱うために必要
#include <pybind11/stl.h>     // C++の標準ライブラリ(vectorなど)を扱うために必要
#include "myrobot.h"

namespace py = pybind11;
using namespace cnoid::vnoid;

// C++側のGym環境を管理するクラス
class VnoidEnv {
public:
    // MuJoCoのデータ
    mjModel* m = NULL;
    mjData* d = NULL;
    // vnoidのロボットクラス
    MyRobot robot;
    double previous_x = 0.0;

    // コンストラクタ：Pythonからモデルファイルのパスを受け取って初期化
    VnoidEnv(const std::string& model_path) {
        char error[1000];
        m = mj_loadXML(model_path.c_str(), 0, error, 1000);
        if (!m) {
            throw std::runtime_error("モデルファイルの読み込みに失敗しました: " + std::string(error));
        }
        d = mj_makeData(m);
        robot.Init(m, d);
    }

    // デストラクタ：MuJoCoのデータを解放
    ~VnoidEnv() {
        if (d) mj_deleteData(d);
        if (m) mj_deleteModel(m);
    }

    // reset関数にもprevious_xの初期化を追加
    py::array_t<double> reset() {
        mj_resetData(m, d);
        mj_forward(m, d);
        previous_x = d->qpos[0]; // ★リセット時のx座標を保存
        return get_observation();
    }

    // Pythonのenv.step()に対応する関数
    py::tuple step(py::array_t<double> action) {
        // 1. Python(Numpy)からのactionをC++のRLParamsに変換
        auto buf = action.request();
        if (buf.ndim != 1 || buf.size < 2) {
            throw std::runtime_error("アクションの次元またはサイズが不正です。");
        }
        double* ptr = static_cast<double*>(buf.ptr);
        RLParams rl_params;
        rl_params.foot_offset.x() = ptr[0];
        rl_params.foot_offset.y() = ptr[1];

        // 2. vnoidの制御サイクルとMuJoCoのシミュレーションを実行
        robot.Control(rl_params);
        mj_step(m, d);

        // 3. 結果をPythonのタプル(obs, reward, terminated, info)で返す
        py::array_t<double> obs = get_observation();
        double reward = compute_reward();
        bool terminated = check_termination();
        
        return py::make_tuple(obs, reward, terminated, py::dict());
    }

private:
    // ★★★これらの中身は後で実装します★★★
    // 観測データを取得してNumpy配列で返す
    py::array_t<double> get_observation() {
    // 観測データの次元数 (qposとqvelの合計)
    const int obs_size = m->nq + m->nv;
    py::array_t<double> obs(obs_size);
    auto obs_ptr = obs.mutable_data();

    // d->qpos と d->qvel から観測データをコピーする
    memcpy(obs_ptr, d->qpos, m->nq * sizeof(double));
    memcpy(obs_ptr + m->nq, d->qvel, m->nv * sizeof(double));

    return obs;
    }

    // 報酬を計算する
    double compute_reward() {
        // 現在のx座標と前のステップのx座標の差分（前進速度）
        double current_x = d->qpos[0];
        double forward_reward = current_x - previous_x;

        // 現在のx座標を次のステップのために保存
        previous_x = current_x;

        // TODO: 生存ボーナスなどを追加しても良い
        double healthy_reward = 1.0; 

        return forward_reward * 5.0 + healthy_reward; // 前進の重みを5倍に
    }

    

    // 終了判定を行う
    bool check_termination() {
        // 胴体のz座標を取得
        double z_position = d->qpos[2];

        // 高さが1.0m～2.0mの範囲外になったら終了
        bool is_terminated = !(z_position >= 1.0 && z_position <= 2.0);

        return is_terminated;
    }
};

// pybind11の魔法：VnoidEnvクラスを "vnoid_rl_env" という名前でPythonに公開
PYBIND11_MODULE(vnoid_rl_env, m) {
    py::class_<VnoidEnv>(m, "VnoidEnv")
        .def(py::init<const std::string&>()) // コンストラクタを公開
        .def("step", &VnoidEnv::step)         // stepメソッドを公開
        .def("reset", &VnoidEnv::reset);      // resetメソッドを公開
}
