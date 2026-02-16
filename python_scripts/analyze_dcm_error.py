#!/usr/bin/env python3
"""
DCM誤差分析スクリプト

ベース部のローカル座標系でのDCM誤差を計算し、
グラフ化・CSV出力を行います。

使用方法:
    python analyze_dcm_error.py [control_log.csv]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os
from pathlib import Path


class QuaternionTransform:
    """クォータニオンを使った座標変換のユーティリティクラス"""
    
    @staticmethod
    def quaternion_conjugate(q):
        """クォータニオンの共役を計算
        
        Args:
            q: クォータニオン [w, x, y, z]
        
        Returns:
            共役クォータニオン [w, -x, -y, -z]
        """
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    @staticmethod
    def quaternion_multiply(q1, q2):
        """クォータニオンの積を計算
        
        Args:
            q1, q2: クォータニオン [w, x, y, z]
        
        Returns:
            積 q1 * q2
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    @staticmethod
    def rotate_vector_by_quaternion(q, v):
        """クォータニオンでベクトルを回転
        
        Args:
            q: クォータニオン [w, x, y, z]
            v: ベクトル [x, y, z]
        
        Returns:
            回転後のベクトル [x, y, z]
        """
        # ベクトルをクォータニオンに変換 (w=0)
        v_quat = np.array([0, v[0], v[1], v[2]])
        
        # q * v * q^*
        q_conj = QuaternionTransform.quaternion_conjugate(q)
        result = QuaternionTransform.quaternion_multiply(
            QuaternionTransform.quaternion_multiply(q, v_quat),
            q_conj
        )
        
        return result[1:4]  # [x, y, z]部分を返す
    
    @staticmethod
    def transform_to_local(pos_global, base_pos, base_ori):
        """グローバル座標をベース部ローカル座標に変換
        
        Args:
            pos_global: グローバル座標 [x, y, z]
            base_pos: ベース部位置 [x, y, z]
            base_ori: ベース部姿勢クォータニオン [w, x, y, z]
        
        Returns:
            ローカル座標 [x, y, z]
        """
        # 1. ベース位置で平行移動
        relative_pos = pos_global - base_pos
        
        # 2. ベース姿勢の共役で回転（グローバル→ローカル）
        q_conj = QuaternionTransform.quaternion_conjugate(base_ori)
        local_pos = QuaternionTransform.rotate_vector_by_quaternion(q_conj, relative_pos)
        
        return local_pos


class DCMErrorAnalyzer:
    """DCM誤差分析クラス"""
    
    def __init__(self, csv_file):
        """
        Args:
            csv_file: control_log.csvのパス
        """
        self.csv_file = csv_file
        self.data = None
        self.results = None
        
    def load_data(self):
        """CSVファイルからデータを読み込む"""
        print(f"📂 データ読み込み中: {self.csv_file}")
        
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"ファイルが見つかりません: {self.csv_file}")
        
        self.data = pd.read_csv(self.csv_file)
        print(f"✅ データ読み込み完了: {len(self.data)} サンプル")
        print(f"   時間範囲: {self.data['time'].min():.2f}秒 ~ {self.data['time'].max():.2f}秒")
        
        # 必要なカラムの存在確認
        required_cols = [
            'time', 
            'base_pos_x', 'base_pos_y', 'base_pos_z',
            'base_pos_ref_x', 'base_pos_ref_y', 'base_pos_ref_z',
            'base_ori_w', 'base_ori_x', 'base_ori_y', 'base_ori_z',
            'base_ori_ref_w', 'base_ori_ref_x', 'base_ori_ref_y', 'base_ori_ref_z',
            'dcm_x', 'dcm_y', 'dcm_z',
            'dcm_ref_x', 'dcm_ref_y', 'dcm_ref_z',
        ]
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"必要なカラムがありません: {missing_cols}")
        
        return self.data
    
    def calculate_dcm_error_in_base_local(self):
        """ベース部ローカル座標系でのDCM誤差を計算
        
        重要: 実測DCMは実際のベース座標系、目標DCMは目標ベース座標系で変換します。
        これにより、各々の基準座標系における相対位置を正確に比較できます。
        """
        print("\n🔄 DCM誤差をベース部ローカル座標系で計算中...")
        
        n_samples = len(self.data)
        
        # 結果を格納する配列
        dcm_actual_local = np.zeros((n_samples, 3))
        dcm_ref_local = np.zeros((n_samples, 3))
        dcm_error_local = np.zeros((n_samples, 3))
        dcm_error_norm = np.zeros(n_samples)
        
        # 各時刻でローカル座標系に変換
        for i in range(n_samples):
            # ベース部の実際の位置と姿勢
            base_pos = np.array([
                self.data.loc[i, 'base_pos_x'],
                self.data.loc[i, 'base_pos_y'],
                self.data.loc[i, 'base_pos_z']
            ])
            
            base_ori = np.array([
                self.data.loc[i, 'base_ori_w'],
                self.data.loc[i, 'base_ori_x'],
                self.data.loc[i, 'base_ori_y'],
                self.data.loc[i, 'base_ori_z']
            ])
            
            # ベース部の目標位置と姿勢
            base_pos_ref = np.array([
                self.data.loc[i, 'base_pos_ref_x'],
                self.data.loc[i, 'base_pos_ref_y'],
                self.data.loc[i, 'base_pos_ref_z']
            ])
            
            base_ori_ref = np.array([
                self.data.loc[i, 'base_ori_ref_w'],
                self.data.loc[i, 'base_ori_ref_x'],
                self.data.loc[i, 'base_ori_ref_y'],
                self.data.loc[i, 'base_ori_ref_z']
            ])
            
            # DCMのグローバル座標
            dcm_actual_global = np.array([
                self.data.loc[i, 'dcm_x'],
                self.data.loc[i, 'dcm_y'],
                self.data.loc[i, 'dcm_z']
            ])
            
            dcm_ref_global = np.array([
                self.data.loc[i, 'dcm_ref_x'],
                self.data.loc[i, 'dcm_ref_y'],
                self.data.loc[i, 'dcm_ref_z']
            ])
            
            # ローカル座標系に変換
            # 実測値は実際のベース座標系で変換
            dcm_actual_local[i] = QuaternionTransform.transform_to_local(
                dcm_actual_global, base_pos, base_ori
            )
            
            # 目標値は目標ベース座標系で変換
            dcm_ref_local[i] = QuaternionTransform.transform_to_local(
                dcm_ref_global, base_pos_ref, base_ori_ref
            )
            
            # 誤差計算
            dcm_error_local[i] = dcm_actual_local[i] - dcm_ref_local[i]
            dcm_error_norm[i] = np.linalg.norm(dcm_error_local[i])
        
        # 結果をDataFrameに格納
        self.results = pd.DataFrame({
            'time': self.data['time'],
            'dcm_actual_local_x': dcm_actual_local[:, 0],
            'dcm_actual_local_y': dcm_actual_local[:, 1],
            'dcm_actual_local_z': dcm_actual_local[:, 2],
            'dcm_ref_local_x': dcm_ref_local[:, 0],
            'dcm_ref_local_y': dcm_ref_local[:, 1],
            'dcm_ref_local_z': dcm_ref_local[:, 2],
            'dcm_error_local_x': dcm_error_local[:, 0],
            'dcm_error_local_y': dcm_error_local[:, 1],
            'dcm_error_local_z': dcm_error_local[:, 2],
            'dcm_error_norm': dcm_error_norm,
        })
        
        # 接触状態も追加（グラフ用）
        if 'obs_contact_right' in self.data.columns:
            self.results['contact_right'] = self.data['obs_contact_right']
        if 'obs_contact_left' in self.data.columns:
            self.results['contact_left'] = self.data['obs_contact_left']
        
        print("✅ 誤差計算完了")
        print(f"   X方向誤差: 平均={np.mean(dcm_error_local[:, 0]):.4f}m, "
              f"RMS={np.sqrt(np.mean(dcm_error_local[:, 0]**2)):.4f}m")
        print(f"   Y方向誤差: 平均={np.mean(dcm_error_local[:, 1]):.4f}m, "
              f"RMS={np.sqrt(np.mean(dcm_error_local[:, 1]**2)):.4f}m")
        print(f"   Z方向誤差: 平均={np.mean(dcm_error_local[:, 2]):.4f}m, "
              f"RMS={np.sqrt(np.mean(dcm_error_local[:, 2]**2)):.4f}m")
        print(f"   誤差ノルム: 平均={np.mean(dcm_error_norm):.4f}m, "
              f"最大={np.max(dcm_error_norm):.4f}m")
        
        return self.results
    
    def save_to_csv(self, output_file='dcm_error_analysis.csv'):
        """分析結果をCSVに保存"""
        if self.results is None:
            raise ValueError("先にcalculate_dcm_error_in_base_local()を実行してください")
        
        print(f"\n💾 結果をCSV保存中: {output_file}")
        self.results.to_csv(output_file, index=False)
        print(f"✅ CSV保存完了: {output_file}")
        
        return output_file
    
    def plot_dcm_error(self, output_pdf='dcm_error_analysis.pdf'):
        """DCM誤差をグラフ化"""
        if self.results is None:
            raise ValueError("先にcalculate_dcm_error_in_base_local()を実行してください")
        
        print(f"\n📊 グラフ生成中: {output_pdf}")
        
        with PdfPages(output_pdf) as pdf:
            # ページ1: DCM誤差の時系列（3パネル）
            self._plot_error_timeseries(pdf)
            
            # ページ2: 誤差ノルムと接触状態
            self._plot_error_norm(pdf)
            
            # ページ3: ローカル座標系でのDCM軌跡（XY平面）
            self._plot_dcm_trajectory_local(pdf)
            
            # ページ4: ローカル座標系でのDCM（XZ平面、YZ平面）
            self._plot_dcm_trajectory_local_xz_yz(pdf)
        
        print(f"✅ グラフ保存完了: {output_pdf}")
        return output_pdf
    
    def _plot_error_timeseries(self, pdf):
        """DCM error time series graph"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('DCM Error Time Series (Base Local Coordinate)', fontsize=16, fontweight='bold')
        
        time = self.results['time']
        
        # X direction error
        axes[0].plot(time, self.results['dcm_error_local_x'], 'r-', linewidth=1.5)
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0].set_ylabel('X Error [m]', fontsize=12)
        axes[0].set_title('X Direction (Forward/Backward)', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Y direction error
        axes[1].plot(time, self.results['dcm_error_local_y'], 'g-', linewidth=1.5)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1].set_ylabel('Y Error [m]', fontsize=12)
        axes[1].set_title('Y Direction (Left/Right)', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Z direction error
        axes[2].plot(time, self.results['dcm_error_local_z'], 'b-', linewidth=1.5)
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[2].set_xlabel('Time [s]', fontsize=12)
        axes[2].set_ylabel('Z Error [m]', fontsize=12)
        axes[2].set_title('Z Direction (Up/Down)', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    def _plot_error_norm(self, pdf):
        """Error norm and contact state graph"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('DCM Error Norm and Contact State', fontsize=16, fontweight='bold')
        
        time = self.results['time']
        
        # Error norm
        axes[0].plot(time, self.results['dcm_error_norm'], 'purple', linewidth=1.5)
        axes[0].set_ylabel('Error Norm [m]', fontsize=12)
        axes[0].set_title('DCM Error Magnitude', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(self.results['dcm_error_norm'])
        max_error = np.max(self.results['dcm_error_norm'])
        axes[0].axhline(y=mean_error, color='r', linestyle='--', alpha=0.5, 
                       label=f'Mean: {mean_error:.4f}m')
        axes[0].legend(fontsize=10)
        
        # Contact state
        if 'contact_right' in self.results.columns and 'contact_left' in self.results.columns:
            axes[1].fill_between(time, 0, self.results['contact_right'], 
                                alpha=0.5, color='red', label='Right Foot Contact')
            axes[1].fill_between(time, -1, -self.results['contact_left'], 
                                alpha=0.5, color='blue', label='Left Foot Contact')
            axes[1].set_xlabel('Time [s]', fontsize=12)
            axes[1].set_ylabel('Contact State', fontsize=12)
            axes[1].set_title('Foot Contact State', fontsize=11)
            axes[1].set_ylim([-1.2, 1.2])
            axes[1].set_yticks([-1, 0, 1])
            axes[1].set_yticklabels(['Left', '', 'Right'])
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    def _plot_dcm_trajectory_local(self, pdf):
        """DCM trajectory in local coordinate system (XY plane)"""
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.suptitle('DCM Trajectory (Base Local Coordinate - XY Plane)', fontsize=16, fontweight='bold')
        
        # Reference trajectory
        ax.plot(self.results['dcm_ref_local_x'], 
               self.results['dcm_ref_local_y'],
               'b--', linewidth=2, alpha=0.7, label='Reference DCM')
        
        # Actual trajectory
        ax.plot(self.results['dcm_actual_local_x'],
               self.results['dcm_actual_local_y'],
               'r-', linewidth=2, alpha=0.7, label='Actual DCM')
        
        # Start and end points
        ax.plot(self.results['dcm_actual_local_x'].iloc[0],
               self.results['dcm_actual_local_y'].iloc[0],
               'go', markersize=12, label='Start')
        ax.plot(self.results['dcm_actual_local_x'].iloc[-1],
               self.results['dcm_actual_local_y'].iloc[-1],
               'rx', markersize=12, label='End')
        
        ax.set_xlabel('X (Forward/Backward) [m]', fontsize=12)
        ax.set_ylabel('Y (Left/Right) [m]', fontsize=12)
        ax.set_title('DCM Motion from Base Frame', fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    def _plot_dcm_trajectory_local_xz_yz(self, pdf):
        """DCM trajectory in local coordinate system (XZ and YZ planes)"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('DCM Trajectory (Base Local Coordinate)', fontsize=16, fontweight='bold')
        
        # XZ plane
        axes[0].plot(self.results['dcm_ref_local_x'],
                    self.results['dcm_ref_local_z'],
                    'b--', linewidth=2, alpha=0.7, label='Reference DCM')
        axes[0].plot(self.results['dcm_actual_local_x'],
                    self.results['dcm_actual_local_z'],
                    'r-', linewidth=2, alpha=0.7, label='Actual DCM')
        axes[0].set_xlabel('X (Forward/Backward) [m]', fontsize=12)
        axes[0].set_ylabel('Z (Up/Down) [m]', fontsize=12)
        axes[0].set_title('XZ Plane', fontsize=11)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')
        
        # YZ plane
        axes[1].plot(self.results['dcm_ref_local_y'],
                    self.results['dcm_ref_local_z'],
                    'b--', linewidth=2, alpha=0.7, label='Reference DCM')
        axes[1].plot(self.results['dcm_actual_local_y'],
                    self.results['dcm_actual_local_z'],
                    'r-', linewidth=2, alpha=0.7, label='Actual DCM')
        axes[1].set_xlabel('Y (Left/Right) [m]', fontsize=12)
        axes[1].set_ylabel('Z (Up/Down) [m]', fontsize=12)
        axes[1].set_title('YZ Plane', fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('equal')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main():
    """メイン処理"""
    # コマンドライン引数からCSVファイル名を取得
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = 'control_log.csv'
    
    print("="*70)
    print("DCM誤差分析スクリプト")
    print("="*70)
    
    # 分析実行
    analyzer = DCMErrorAnalyzer(csv_file)
    
    try:
        # データ読み込み
        analyzer.load_data()
        
        # DCM誤差計算
        analyzer.calculate_dcm_error_in_base_local()
        
        # CSV出力
        analyzer.save_to_csv('dcm_error_analysis.csv')
        
        # グラフ生成
        analyzer.plot_dcm_error('dcm_error_analysis.pdf')
        
        print("\n" + "="*70)
        print("✅ すべての処理が完了しました")
        print("="*70)
        print(f"📄 出力ファイル:")
        print(f"   - dcm_error_analysis.csv")
        print(f"   - dcm_error_analysis.pdf")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
