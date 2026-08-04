#include "vnoid_env.h"

// ★★★ sample_controller_mujocoと同じグローバル変数（インタラクション用） ★★★
// これらは1つのVnoidEnvインスタンスでのみ使用される
static bool button_left = false;
static bool button_middle = false;
static bool button_right = false;
static double lastx = 0;
static double lasty = 0;

// ★★★ sample_controller_mujocoと同じGLFW初期化 ★★★
void VnoidEnv::initializeGLFW() {
    // init GLFW (sample_controller_mujocoと同じ)
    if (!glfwInit()) {
        throw std::runtime_error("Could not initialize GLFW");
    }
    glfw_initialized = true;

    // ★★★ sample_controller_mujocoと同じ可視ウィンドウ作成 ★★★
    window = glfwCreateWindow(1200, 900, "VnoidEnv Interactive", NULL, NULL);
    
    if (!window) {
        glfwTerminate();
        glfw_initialized = false;
        throw std::runtime_error("GLFWウィンドウの作成に失敗しました。");
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // ★★★ sample_controller_mujocoと同じコールバック設定 ★★★
    glfwSetWindowUserPointer(window, this);  // thisポインタを保存
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);
}

// ★★★ sample_controller_mujocoと同じレンダリング初期化 ★★★
void VnoidEnv::initializeRenderer() {
    // ★★★ sample_controller_mujocoと同じ初期化 ★★★
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    // debug用
    // opt.flags[mjVIS_CONTACTPOINT] = 1;
    // opt.flags[mjVIS_CONTACTFORCE] = 1;
    opt.sitegroup[1] = 1;  // r_foot_marker / l_foot_marker (group=1)
    mjv_defaultScene(&scn);     // ← これが足りない！
    mjr_defaultContext(&con);
    
    // create scene and context (sample_controller_mujocoと同じ)
    mjv_makeScene(m, &scn, 2000);
    scene_initialized = true;
    
    mjr_makeContext(m, &con, mjFONTSCALE_150);
    context_initialized = true;
}

// ★★★ sample_controller_mujocoと同じ表示更新 ★★★
void VnoidEnv::updateDisplay() {
    if (!rendering_enabled || !window) return;

    // get framebuffer viewport (sample_controller_mujocoと同じ)
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // update scene and render (sample_controller_mujocoと同じ)
    mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    // swap OpenGL buffers (sample_controller_mujocoと同じ)
    glfwSwapBuffers(window);

    // process pending GUI events (sample_controller_mujocoと同じ)
    glfwPollEvents();
}

// ★★★ 録画用レンダリング（画面表示とは別） ★★★
py::array_t<unsigned char> VnoidEnv::render() {
    if (!rendering_enabled || !scene_initialized || !context_initialized) {
        throw std::runtime_error("レンダリングが無効化されています。");
    }
    
    try {

        // // カメラをロボットに追従させる、必要ないときは消せ
        cam.lookat[0] = d->qpos[0];  // X座標
        cam.lookat[1] = d->qpos[1];  // Y座標
        cam.lookat[2] = d->qpos[2];  // Z座標


        // 録画用の固定ビューポート
        mjrRect viewport = {0, 0, 1280, 720};
        
        // シーン更新・レンダリング
        mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);
        
        // ピクセルデータを読み出し
        size_t buffer_size = viewport.width * viewport.height * 3;
        auto buffer = std::make_unique<unsigned char[]>(buffer_size);
        mjr_readPixels(buffer.get(), nullptr, viewport, &con);
        
        // Python配列に変換
        py::capsule free_when_done(buffer.release(), [](void *f) {
            delete[] static_cast<unsigned char *>(f);
        });
        
        return py::array_t<unsigned char>(
            {viewport.height, viewport.width, 3},
            {viewport.width * 3, 3, 1},
            static_cast<unsigned char*>(free_when_done.get_pointer()),
            free_when_done
        );
    } catch (const std::exception& e) {
        throw std::runtime_error("レンダリング中にエラーが発生しました: " + std::string(e.what()));
    }
}

// GLFWコールバック関数
void VnoidEnv::keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
        VnoidEnv* env = static_cast<VnoidEnv*>(glfwGetWindowUserPointer(window));
        if (env) {
            env->reset();
        }
    }
}

void VnoidEnv::mouse_button(GLFWwindow* window, int button, int act, int mods) {
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    glfwGetCursorPos(window, &lastx, &lasty);
}

void VnoidEnv::mouse_move(GLFWwindow* window, double xpos, double ypos) {
    VnoidEnv* env = static_cast<VnoidEnv*>(glfwGetWindowUserPointer(window));
    if (!env || !button_left && !button_middle && !button_right) return;

    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    int width, height;
    glfwGetWindowSize(window, &width, &height);

    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    mjtMouse action;
    if (button_right) {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    } else if (button_left) {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    } else {
        action = mjMOUSE_ZOOM;
    }

    mjv_moveCamera(env->GetModel(), action, dx/height, dy/height, env->GetScene(), env->GetCamera());
}

void VnoidEnv::scroll(GLFWwindow* window, double xoffset, double yoffset) {
    VnoidEnv* env = static_cast<VnoidEnv*>(glfwGetWindowUserPointer(window));
    if (!env) return;
    mjv_moveCamera(env->GetModel(), mjMOUSE_ZOOM, 0, -0.05*yoffset, env->GetScene(), env->GetCamera());
}
