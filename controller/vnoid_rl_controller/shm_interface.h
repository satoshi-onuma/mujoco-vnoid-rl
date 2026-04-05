/**
 * @file shm_interface.h
 * @brief Python (強化学習エージェント) と C++ (Choreonoid シミュレータ) を
 *        POSIX 共有メモリで繋ぐプロセス間通信インターフェース。
 *
 * ## システム概要
 *
 *   Python プロセス                      C++ プロセス (Choreonoid)
 *   ─────────────────────                ───────────────────────────
 *   RLlib / Gymnasium 環境               VnoidRLController
 *   ChoreonoidShmClient                  ShmInterface (create=false)
 *        │                                     │
 *        └──── POSIX 共有メモリ (/dev/shm) ─────┘
 *              /vnoid_rl_shm_<env_id>
 *
 * ## 共有メモリレイアウト (合計 824 bytes)
 *
 *   offset   0 │ ControlBlock (24 bytes)  ← 制御フラグ・報酬
 *   offset  24 │ ActionBlock  (232 bytes) ← Python → C++: 関節目標角度
 *   offset 256 │ StateBlock   (568 bytes) ← C++ → Python: 観測ベクトル
 *
 * ## 同期プロトコル (1 ステップ)
 *
 *   Python                        C++ (control() ループ)
 *   ────────────────────          ─────────────────────────────────
 *   1. ready = 0          →
 *   2. action を書き込み  →
 *   3. step_request = 1   →       4. step_request == 1 を検知
 *                                 5. action 読み取り → PD 制御実行
 *                                 6. obs / reward / done を書き込み
 *                                 7. step_request = 0
 *                  ←              8. ready = 1  (set_ready は最後)
 *   9. ready == 1 を確認
 *  10. obs / reward / done を読み取り
 *
 *   ※ Python が手順1 (ready=0) を省くと、前回の ready=1 が残っていて
 *      C++ の処理完了前に即リターンしてしまう (旧バグ)。
 *
 * ## アライメントについて
 *
 *   ControlBlock は volatile float を含む旧設計では 20 bytes だった。
 *   直後の ActionBlock は double[] を持つため 8 byte 境界が必要だが、
 *   20 % 8 != 0 なのでコンパイラが 4 bytes の暗黙 padding を挿入し、
 *   Python 側のオフセット計算と不一致が起きていた。
 *   現在は int _pad を明示追加して 24 bytes に揃えている。
 *
 * ## 並列環境
 *
 *   ENV_ID 環境変数でインスタンスを識別する。
 *   n 個並列実行する場合、Choreonoid を ENV_ID=0..n-1 で別々に起動する。
 */
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

// ---------------------------------------------------------------------------
// 次元数定数
// ---------------------------------------------------------------------------

/// ロボットの関節数 (行動空間の次元数)
constexpr size_t NUM_JOINTS = 29;

/**
 * 観測ベクトルの次元数 (71)
 *   [  0- 2]  base_pos    : ルートリンク位置 (x, y, z)
 *   [  3- 6]  ori_quat    : ルートリンク姿勢クォータニオン (w, x, y, z)
 *   [  7- 9]  base_vel    : ルートリンク線速度 (x, y, z)
 *   [ 10-12]  base_angvel : ルートリンク角速度 (x, y, z)
 *   [ 13-41]  joint_q     : 全関節角度 (29 joints)
 *   [ 42-70]  joint_dq    : 全関節角速度 (29 joints)
 */
constexpr size_t NUM_OBSERVATIONS = 71;

// ---------------------------------------------------------------------------
// 共有メモリ構造体
// ---------------------------------------------------------------------------

/**
 * @brief 制御フラグ・報酬を格納するブロック (24 bytes)
 *
 * フィールドは int で統一し、__atomic_* 組み込み関数でアクセスする。
 * reward_bits は float の IEEE 754 ビット表現を int に memcpy して格納する。
 * _pad は後続の ActionBlock (double[]) が 8 byte 境界に揃うよう補填する。
 *
 * メモリ上の並び:
 *   byte  0- 3 : step_request
 *   byte  4- 7 : reset_request
 *   byte  8-11 : done
 *   byte 12-15 : ready
 *   byte 16-19 : reward_bits
 *   byte 20-23 : _pad  (= 0、使用しない)
 */
struct ControlBlock {
    int step_request;   ///< Python → C++: 1 = step 実行要求
    int reset_request;  ///< Python → C++: 1 = reset 要求
    int done;           ///< C++ → Python: 1 = エピソード終了
    int ready;          ///< C++ → Python: 1 = 処理完了 (Python は次のリクエスト前に 0 にする)
    int reward_bits;    ///< C++ → Python: 報酬値 (float のビット表現、memcpy で変換)
    int _pad;           ///< ActionBlock の 8 byte アライメント用パディング (未使用)
};

/// 行動ブロック: Python → C++ (232 bytes)
struct ActionBlock {
    double joint_targets[NUM_JOINTS];  ///< 全関節の目標角度 (正規化: -1〜1)
};

/// 観測ブロック: C++ → Python (568 bytes)
struct StateBlock {
    double observations[NUM_OBSERVATIONS];  ///< 観測ベクトル (71 次元、内訳は NUM_OBSERVATIONS を参照)
};

/**
 * @brief 共有メモリ全体のレイアウト (合計 824 bytes)
 *
 *   offset   0 : ControlBlock (24 bytes)
 *   offset  24 : ActionBlock  (232 bytes)
 *   offset 256 : StateBlock   (568 bytes)
 */
struct SharedMemory {
    ControlBlock control;  ///< offset   0
    ActionBlock  action;   ///< offset  24
    StateBlock   state;    ///< offset 256
};

// コンパイル時にレイアウトが Python 側の計算と一致するか検証する。
// ミスマッチが検出された場合はビルドが失敗するため、実行時に無言でずれることはない。
static_assert(sizeof(ControlBlock) == 24,           "ControlBlock size mismatch");
static_assert(offsetof(SharedMemory, action) == 24, "ActionBlock offset mismatch");
static_assert(offsetof(SharedMemory, state) == 256, "StateBlock offset mismatch");

// ---------------------------------------------------------------------------
// ShmInterface
// ---------------------------------------------------------------------------

/**
 * @brief POSIX 共有メモリの生成・マップ・解放を管理するクラス。
 *
 * Python 側 (create=true) が先に共有メモリを作成し、
 * C++ 側 (create=false) が既存の共有メモリに接続する。
 * デストラクタはオーナー (create=true) のみ shm_unlink を呼ぶ。
 */
class ShmInterface {
private:
    std::string   shm_name_;   ///< /vnoid_rl_shm_<env_id>
    int           shm_fd_;
    SharedMemory* shm_ptr_;
    bool          is_owner_;   ///< true のとき、デストラクタで shm_unlink する

public:
    /**
     * @param env_id  環境 ID (ENV_ID 環境変数と対応)
     * @param create  true: 共有メモリを新規作成 (Python 側), false: 既存に接続 (C++ 側)
     */
    ShmInterface(int env_id, bool create = false)
        : shm_fd_(-1), shm_ptr_(nullptr), is_owner_(create)
    {
        shm_name_ = "/vnoid_rl_shm_" + std::to_string(env_id);

        if (create) {
            shm_unlink(shm_name_.c_str());  // 残存していればクリーンアップ

            shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
            if (shm_fd_ == -1)
                throw std::runtime_error("Failed to create shared memory: " + shm_name_);

            if (ftruncate(shm_fd_, sizeof(SharedMemory)) == -1) {
                close(shm_fd_);
                throw std::runtime_error("Failed to set shared memory size");
            }
        } else {
            shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR, 0666);
            if (shm_fd_ == -1)
                throw std::runtime_error("Failed to open shared memory: " + shm_name_);
        }

        shm_ptr_ = static_cast<SharedMemory*>(
            mmap(nullptr, sizeof(SharedMemory), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0)
        );
        if (shm_ptr_ == MAP_FAILED) {
            close(shm_fd_);
            throw std::runtime_error("Failed to map shared memory");
        }

        if (create)
            std::memset(shm_ptr_, 0, sizeof(SharedMemory));
    }

    ~ShmInterface() {
        if (shm_ptr_ && shm_ptr_ != MAP_FAILED)
            munmap(shm_ptr_, sizeof(SharedMemory));
        if (shm_fd_ != -1)
            close(shm_fd_);
        if (is_owner_)
            shm_unlink(shm_name_.c_str());
    }

    // --- 生ポインタアクセス (直接書き込みが必要な場合) ---
    SharedMemory*       get()       { return shm_ptr_; }
    const SharedMemory* get() const { return shm_ptr_; }

    ControlBlock& control() { return shm_ptr_->control; }
    ActionBlock&  action()  { return shm_ptr_->action;  }
    StateBlock&   state()   { return shm_ptr_->state;   }

    // --- C++ → Python 通知メソッド ---

    /**
     * @brief 処理完了を Python に通知する。
     *
     * obs / reward / done の書き込みが CPU のリオーダリングにより
     * ready=1 の書き込みより後に見えると Python が古いデータを読む。
     * __sync_synchronize() で全ストアを先にコミットしてから ready=1 をセットする。
     */
    void set_ready(bool ready) {
        __sync_synchronize();
        __atomic_store_n(&shm_ptr_->control.ready, ready ? 1 : 0, __ATOMIC_SEQ_CST);
    }

    /// エピソード終了フラグをセット
    void set_done(bool done) {
        __atomic_store_n(&shm_ptr_->control.done, done ? 1 : 0, __ATOMIC_SEQ_CST);
    }

    /**
     * @brief 報酬値をセット。
     *
     * float を直接書くと __atomic_store_n が使えないため、
     * memcpy で同サイズの int に変換してアトミックに書き込む。
     */
    void set_reward(float reward) {
        int bits;
        std::memcpy(&bits, &reward, sizeof(float));
        __atomic_store_n(&shm_ptr_->control.reward_bits, bits, __ATOMIC_SEQ_CST);
    }

    // --- Python → C++ 要求確認メソッド ---

    /// Python から step 実行要求が来ているか
    bool has_step_request() const {
        return __atomic_load_n(&shm_ptr_->control.step_request, __ATOMIC_SEQ_CST) == 1;
    }

    /// Python から reset 要求が来ているか
    bool has_reset_request() const {
        return __atomic_load_n(&shm_ptr_->control.reset_request, __ATOMIC_SEQ_CST) == 1;
    }

    /// step_request フラグを 0 にクリアする
    void clear_step_request() {
        __atomic_store_n(&shm_ptr_->control.step_request, 0, __ATOMIC_SEQ_CST);
    }

    /// reset_request フラグを 0 にクリアする
    void clear_reset_request() {
        __atomic_store_n(&shm_ptr_->control.reset_request, 0, __ATOMIC_SEQ_CST);
    }

    // --- ユーティリティ ---

    /**
     * @brief 環境 ID を ENV_ID 環境変数から取得する。
     * @return ENV_ID が設定されていればその値、未設定なら 0
     */
    static int get_env_id() {
        const char* s = std::getenv("ENV_ID");
        return s ? std::atoi(s) : 0;
    }
};

}  // namespace vnoid_rl
