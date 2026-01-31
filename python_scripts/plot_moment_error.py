#!/usr/bin/env python3
"""
回復モーメントの2乗誤差和を計算・可視化するスクリプト

CSVファイルから回復モーメントのx,y方向の誤差を読み込み、
2乗誤差和（SSE: Sum of Squared Errors）を計算してプロットします。
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def calculate_moment_error_metrics(csv_file):
    """
    回復モーメントの誤差メトリクスを計算
    
    Parameters:
    -----------
    csv_file : str
        制御ログのCSVファイルパス
    
    Returns:
    --------
    dict : 各種メトリクス
    """
    # CSVファイル読み込み
    df = pd.read_csv(csv_file)
    
    # 必要なカラムの存在確認
    required_cols = ['time', 'moment_diff_x', 'moment_diff_y']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"必要なカラムが見つかりません: {missing_cols}")
    
    # 2乗誤差を計算
    df['moment_error_x_squared'] = df['moment_diff_x'] ** 2
    df['moment_error_y_squared'] = df['moment_diff_y'] ** 2
    
    # x,y方向の2乗誤差和（各時刻での和）
    df['moment_sse_xy'] = df['moment_error_x_squared'] + df['moment_error_y_squared']
    
    # 2乗平均平方根誤差（RMSE）
    rmse_x = np.sqrt(df['moment_error_x_squared'].mean())
    rmse_y = np.sqrt(df['moment_error_y_squared'].mean())
    rmse_xy = np.sqrt(df['moment_sse_xy'].mean())
    
    # 全期間の2乗誤差総和
    total_sse_x = df['moment_error_x_squared'].sum()
    total_sse_y = df['moment_error_y_squared'].sum()
    total_sse_xy = df['moment_sse_xy'].sum()
    
    metrics = {
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'rmse_xy': rmse_xy,
        'total_sse_x': total_sse_x,
        'total_sse_y': total_sse_y,
        'total_sse_xy': total_sse_xy,
    }
    
    return df, metrics


def plot_moment_errors(df, metrics, output_file=None):
    """
    回復モーメントの誤差をプロット
    
    Parameters:
    -----------
    df : pd.DataFrame
        制御ログデータ
    metrics : dict
        計算されたメトリクス
    output_file : str, optional
        保存するファイル名
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. x,y方向の誤差の時間変化
    ax1 = axes[0]
    ax1.plot(df['time'], df['moment_diff_x'], label='X方向誤差', alpha=0.7)
    ax1.plot(df['time'], df['moment_diff_y'], label='Y方向誤差', alpha=0.7)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('時間 [s]')
    ax1.set_ylabel('モーメント誤差 [Nm]')
    ax1.set_title('回復モーメント誤差（実測 - 所望値）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 統計情報を表示
    textstr = f'RMSE_x: {metrics["rmse_x"]:.4f} Nm\n'
    textstr += f'RMSE_y: {metrics["rmse_y"]:.4f} Nm'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. 2乗誤差の時間変化
    ax2 = axes[1]
    ax2.plot(df['time'], df['moment_error_x_squared'], 
             label='X方向2乗誤差', alpha=0.7)
    ax2.plot(df['time'], df['moment_error_y_squared'], 
             label='Y方向2乗誤差', alpha=0.7)
    ax2.set_xlabel('時間 [s]')
    ax2.set_ylabel('2乗誤差 [Nm²]')
    ax2.set_title('回復モーメント2乗誤差')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. x,y方向の2乗誤差和（SSE）
    ax3 = axes[2]
    ax3.plot(df['time'], df['moment_sse_xy'], 
             label='X+Y 2乗誤差和', color='red', linewidth=1.5)
    ax3.fill_between(df['time'], 0, df['moment_sse_xy'], 
                      alpha=0.3, color='red')
    ax3.set_xlabel('時間 [s]')
    ax3.set_ylabel('2乗誤差和 [Nm²]')
    ax3.set_title('X・Y方向の回復モーメント2乗誤差和')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 累積SSEを表示
    textstr = f'RMSE (X+Y): {metrics["rmse_xy"]:.4f} Nm\n'
    textstr += f'総2乗誤差和: {metrics["total_sse_xy"]:.4f} Nm²'
    ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ グラフを保存しました: {output_file}")
    
    plt.show()


def print_metrics_summary(metrics):
    """メトリクスのサマリーを表示"""
    print("\n" + "="*60)
    print("回復モーメント誤差の統計サマリー")
    print("="*60)
    print(f"\n【2乗平均平方根誤差 (RMSE)】")
    print(f"  X方向: {metrics['rmse_x']:.6f} Nm")
    print(f"  Y方向: {metrics['rmse_y']:.6f} Nm")
    print(f"  X+Y:   {metrics['rmse_xy']:.6f} Nm")
    
    print(f"\n【総2乗誤差和 (Total SSE)】")
    print(f"  X方向: {metrics['total_sse_x']:.6f} Nm²")
    print(f"  Y方向: {metrics['total_sse_y']:.6f} Nm²")
    print(f"  X+Y:   {metrics['total_sse_xy']:.6f} Nm²")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='回復モーメントの2乗誤差和を計算・可視化'
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        default='control_log.csv',
        help='制御ログのCSVファイルパス（デフォルト: control_log.csv）'
    )
    parser.add_argument(
        '-o', '--output',
        default='moment_error_analysis.pdf',
        help='出力ファイル名（デフォルト: moment_error_analysis.pdf）'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='グラフを表示せず、統計情報のみ出力'
    )
    
    args = parser.parse_args()
    
    # ファイルの存在確認
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"❌ エラー: ファイルが見つかりません: {args.csv_file}")
        return 1
    
    try:
        # データ読み込みと計算
        print(f"📊 CSVファイルを読み込んでいます: {args.csv_file}")
        df, metrics = calculate_moment_error_metrics(args.csv_file)
        
        # 統計情報を表示
        print_metrics_summary(metrics)
        
        # グラフ作成
        if not args.no_plot:
            plot_moment_errors(df, metrics, args.output)
        
        return 0
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
