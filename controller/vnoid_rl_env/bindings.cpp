#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>   // PythonのNumpy配列を扱うために必要
#include <pybind11/stl.h>     // C++の標準ライブラリ(vectorなど)を扱うために必要
#include "myrobot.h"
#include <GLFW/glfw3.h> // ★ GLFWを追加
#include <mujoco/mujoco.h> // mjv/mjr関連の定義のために必要
#include <mutex> // ★ mutexを追加

namespace py = pybind11;
using namespace cnoid::vnoid;


// ★★★ GLFWの初期化を管理するための静的カウンタとミューテックスを追加 ★★★
static int g_glfw_init_count = 0;
static std::mutex g_glfw_mutex;

// C++側のGym環境を管理するクラス
class VnoidEnv {
public:
    // MuJoCoのデータ
    mjModel* m = NULL;
    mjData* d = NULL;
    // vnoidのロボットクラス
    MyRobot robot;
    double previous_x = 0.0;

    // ★★★ レンダリング用のオブジェクトを追加 ★★★
    mjvCamera cam;
    mjvOption opt;
    mjvScene scn;
    mjrContext con;
    // ビューポート（画像の解像度）
    mjrRect viewport;
    GLFWwindow* window = nullptr;

    // コンストラクタ：Pythonからモデルファイルのパスを受け取って初期化
    VnoidEnv(const std::string& model_path) {
       // ★ スレッドセーフなGLFW初期化処理
        std::lock_guard<std::mutex> lock(g_glfw_mutex);
        if (g_glfw_init_count++ == 0) {
            if (!glfwInit()) {
                throw std::runtime_error("GLFWの初期化に失敗しました。");
            }
        }
        
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        window = glfwCreateWindow(640, 480, "Headless MuJoCo", NULL, NULL);
        if (!window) {
            // glfwInitが成功したのにウィンドウ作成が失敗した場合のクリーンアップ
            if (--g_glfw_init_count == 0) {
                glfwTerminate();
            }
            throw std::runtime_error("GLFWウィンドウの作成に失敗しました。");
        }
        glfwMakeContextCurrent(window);
        
        char error[1000];
        m = mj_loadXML(model_path.c_str(), 0, error, 1000);
        if (!m) {

            // 例外を投げる前にリソースを解放
            glfwDestroyWindow(window);
            if (--g_glfw_init_count == 0) {
                glfwTerminate();
            }
            throw std::runtime_error("モデルファイルの読み込みに失敗しました: " + std::string(error));
        }
        d = mj_makeData(m);
        robot.Init(m, d);

        // ★★★ レンダリング関連の初期化を追加 ★★★
        mjv_defaultCamera(&cam);
        mjv_defaultOption(&opt);
        mjr_defaultContext(&con);
        mjv_makeScene(m, &scn, 2000);
        mjr_makeContext(m, &con, mjFONTSCALE_150);

        // 録画する画像の解像度を設定 (例: 640x480)
        viewport = {0, 0, 640, 480};

        // ★★★ この行を追加して、nqとnvの値を確認する ★★★
        printf("DEBUG INFO: nq = %d, nv = %d, Total Obs Size = %d\n", m->nq, m->nv, m->nq + m->nv);
    }

    // デストラクタ：MuJoCoのデータを解放
    ~VnoidEnv() {
        // ★★★ レンダリング関連の解放処理を追加 ★★★
        mjv_freeScene(&scn);
        mjr_freeContext(&con);
        
        if (d) mj_deleteData(d);
        if (m) mj_deleteModel(m);

       // ★ スレッドセーフなGLFW終了処理
        std::lock_guard<std::mutex> lock(g_glfw_mutex);
        if (window) {
            glfwDestroyWindow(window);
        }
        if (--g_glfw_init_count == 0) {
            glfwTerminate();
        }
    }

    // ★★★ 以下の4行を追加して、コピーとムーブを禁止する ★★★
    VnoidEnv(const VnoidEnv&) = delete;
    VnoidEnv& operator=(const VnoidEnv&) = delete;
    VnoidEnv(VnoidEnv&&) = delete;
    VnoidEnv& operator=(VnoidEnv&&) = delete;

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

     // ★★★ render関数を実装 ★★★
    py::array_t<unsigned char> render() {
        // 1. シーンデータを更新
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);

        // 2. シーンを描画バッファにレンダリング
        mjr_render(viewport, &scn, &con);

        // 3. 描画バッファからピクセルデータを読み出す
        auto buffer = new unsigned char[viewport.width * viewport.height * 3];
        mjr_readPixels(buffer, NULL, viewport, &con);

        // 4. pybind11でPython(Numpy)配列に変換して返す
        //    (Python側でメモリを管理するように設定)
        py::capsule free_when_done(buffer, [](void *f) {
            delete[] static_cast<unsigned char *>(f);
        });
        
        return py::array_t<unsigned char>(
            {viewport.height, viewport.width, 3}, // shape (高さ, 幅, 3ch)
            {viewport.width * 3, 3, 1},            // strides
            buffer,                                // buffer pointer
            free_when_done);
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
        double hips_z_position = d->qpos[2];

        // HIPSの高さが0.5m未満になったら転倒とみなす
        bool is_terminated = (hips_z_position < 0.5);

        return is_terminated;
    }

   
};

// pybind11の魔法：VnoidEnvクラスを "vnoid_rl_env" という名前でPythonに公開
PYBIND11_MODULE(vnoid_rl_env, m) {
    py::class_<VnoidEnv>(m, "VnoidEnv")
        .def(py::init<const std::string&>()) // コンストラクタを公開
        .def("step", &VnoidEnv::step)         // stepメソッドを公開
        .def("reset", &VnoidEnv::reset)     // resetメソッドを公開
        .def("render", &VnoidEnv::render); // ★ renderメソッドを公開
}
