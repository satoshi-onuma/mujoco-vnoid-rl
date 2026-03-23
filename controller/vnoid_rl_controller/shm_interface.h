#pragma once

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <string>
#include <stdexcept>
#include <cstdlib>

namespace vnoid_rl {

// 共有メモリのサイズ定義
constexpr size_t NUM_JOINTS = 29;
constexpr size_t NUM_OBSERVATIONS = 73;

// 制御ブロック：Python ←→ C++ 間の制御信号
struct ControlBlock {
    volatile int step_request;   // Python → C++: 1 = step実行要求
    volatile int reset_request;  // Python → C++: 1 = reset要求
    volatile int done;           // C++ → Python: 1 = エピソード終了
    volatile int ready;          // C++ → Python: 1 = 準備完了
    volatile float reward;       // C++ → Python: 報酬値
};

// 行動ブロック：Python → C++
struct ActionBlock {
    double joint_targets[NUM_JOINTS];  // 全関節の目標角度（正規化: -1〜1）
};

// 観測ブロック：C++ → Python
struct StateBlock {
    double observations[NUM_OBSERVATIONS];  // 観測ベクトル（73次元）
};

// 共有メモリ全体の構造
struct SharedMemory {
    ControlBlock control;
    ActionBlock action;
    StateBlock state;
};

// 共有メモリマネージャクラス
class ShmInterface {
private:
    std::string shm_name_;
    int shm_fd_;
    SharedMemory* shm_ptr_;
    bool is_owner_;

public:
    ShmInterface(int env_id, bool create = false) 
        : shm_fd_(-1), shm_ptr_(nullptr), is_owner_(create) {
        
        // 共有メモリ名を環境IDから生成
        shm_name_ = "/vnoid_rl_shm_" + std::to_string(env_id);
        
        if (create) {
            // 既存の共有メモリを削除（クリーンアップ）
            shm_unlink(shm_name_.c_str());
            
            // 共有メモリを作成
            shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
            if (shm_fd_ == -1) {
                throw std::runtime_error("Failed to create shared memory: " + shm_name_);
            }
            
            // サイズを設定
            if (ftruncate(shm_fd_, sizeof(SharedMemory)) == -1) {
                close(shm_fd_);
                throw std::runtime_error("Failed to set shared memory size");
            }
        } else {
            // 既存の共有メモリを開く
            shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR, 0666);
            if (shm_fd_ == -1) {
                throw std::runtime_error("Failed to open shared memory: " + shm_name_);
            }
        }
        
        // メモリマップ
        shm_ptr_ = static_cast<SharedMemory*>(
            mmap(nullptr, sizeof(SharedMemory), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0)
        );
        
        if (shm_ptr_ == MAP_FAILED) {
            close(shm_fd_);
            throw std::runtime_error("Failed to map shared memory");
        }
        
        // 作成時は初期化
        if (create) {
            std::memset(shm_ptr_, 0, sizeof(SharedMemory));
        }
    }
    
    ~ShmInterface() {
        if (shm_ptr_ != nullptr && shm_ptr_ != MAP_FAILED) {
            munmap(shm_ptr_, sizeof(SharedMemory));
        }
        if (shm_fd_ != -1) {
            close(shm_fd_);
        }
        if (is_owner_) {
            shm_unlink(shm_name_.c_str());
        }
    }
    
    // アクセサメソッド
    SharedMemory* get() { return shm_ptr_; }
    const SharedMemory* get() const { return shm_ptr_; }
    
    ControlBlock& control() { return shm_ptr_->control; }
    ActionBlock& action() { return shm_ptr_->action; }
    StateBlock& state() { return shm_ptr_->state; }
    
    // ヘルパーメソッド
    void set_ready(bool ready) {
        shm_ptr_->control.ready = ready ? 1 : 0;
    }
    
    void set_done(bool done) {
        shm_ptr_->control.done = done ? 1 : 0;
    }
    
    void set_reward(float reward) {
        shm_ptr_->control.reward = reward;
    }
    
    bool has_step_request() const {
        return shm_ptr_->control.step_request == 1;
    }
    
    bool has_reset_request() const {
        return shm_ptr_->control.reset_request == 1;
    }
    
    void clear_step_request() {
        shm_ptr_->control.step_request = 0;
    }
    
    void clear_reset_request() {
        shm_ptr_->control.reset_request = 0;
    }
    
    // 環境IDを環境変数から取得
    static int get_env_id() {
        const char* env_id_str = std::getenv("ENV_ID");
        if (env_id_str == nullptr) {
            return 0;  // デフォルトは0
        }
        return std::atoi(env_id_str);
    }
};

}  // namespace vnoid_rl
