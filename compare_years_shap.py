#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
年度間SHAP値比較ツール

複数年度のSHAP値を比較して、特徴量の重要度変化を分析します。

使用例:
    # 東京芝中長距離の2021-2023年を比較
    python compare_years_shap.py --model tokyo_turf_3ageup_long --years 2021 2022 2023
    
    # 阪神ダート短距離の2022-2024年を比較
    python compare_years_shap.py --model hanshin_dirt_3ageup_short --years 2022 2023 2024
    
    # 出力先ディレクトリをカスタマイズ
    python compare_years_shap.py --model tokyo_turf_3ageup_long --years 2021 2022 2023 --output shap_year_comparison
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from scipy.stats import pearsonr, spearmanr

# 日本語フォント設定
rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def load_shap_csv(model_name, year, base_dir='shap_analysis'):
    """
    指定されたモデルと年度のSHAP CSVを読み込む
    
    Args:
        model_name (str): モデル名
        year (int): 年度
        base_dir (str): SHAP分析結果のベースディレクトリ
        
    Returns:
        pd.DataFrame or None: SHAP重要度データフレーム（読み込み失敗時はNone）
    """
    csv_path = Path(base_dir) / f"{model_name}_importance.csv"
    
    if not csv_path.exists():
        print(f"⚠️  SHAP CSVが見つかりません: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        df['year'] = year  # 年度カラム追加
        print(f"✅ {year}年データ読み込み成功: {len(df)} features")
        return df
    except Exception as e:
        print(f"❌ {year}年データ読み込みエラー: {e}")
        return None


def calculate_year_correlation(df1, df2, year1, year2):
    """
    2年度間のSHAP値相関を計算
    
    Args:
        df1 (pd.DataFrame): 年度1のSHAPデータ
        df2 (pd.DataFrame): 年度2のSHAPデータ
        year1 (int): 年度1
        year2 (int): 年度2
        
    Returns:
        dict: 相関統計情報
    """
    # 共通特徴量の抽出
    common_features = set(df1['feature'].values) & set(df2['feature'].values)
    
    if not common_features:
        return {
            'year1': year1,
            'year2': year2,
            'common_features': 0,
            'pearson_r': np.nan,
            'spearman_r': np.nan
        }
    
    # 共通特徴量でソートしてデータ取得
    sorted_features = sorted(common_features)
    
    df1_filtered = df1[df1['feature'].isin(sorted_features)].set_index('feature').loc[sorted_features]
    df2_filtered = df2[df2['feature'].isin(sorted_features)].set_index('feature').loc[sorted_features]
    
    # Pearson相関（線形相関）
    pearson_r, pearson_p = pearsonr(df1_filtered['mean_abs_shap'], df2_filtered['mean_abs_shap'])
    
    # Spearman相関（順位相関）
    spearman_r, spearman_p = spearmanr(df1_filtered['mean_abs_shap'], df2_filtered['mean_abs_shap'])
    
    return {
        'year1': year1,
        'year2': year2,
        'common_features': len(common_features),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p
    }


def plot_year_comparison(dfs_dict, model_name, output_dir, top_n=20):
    """
    年度間SHAP値比較プロットを生成
    
    Args:
        dfs_dict (dict): {year: DataFrame} 形式の辞書
        model_name (str): モデル名
        output_dir (Path): 出力ディレクトリ
        top_n (int): 表示する上位特徴量数
    """
    years = sorted(dfs_dict.keys())
    
    if len(years) < 2:
        print("⚠️  比較には最低2年度のデータが必要です")
        return
    
    # 1. 上位特徴量の年度間比較（棒グラフ）
    fig, axes = plt.subplots(1, len(years), figsize=(6*len(years), 8), sharey=True)
    
    if len(years) == 1:
        axes = [axes]
    
    # 全年度で共通の上位特徴量を抽出（最初の年度基準）
    base_year = years[0]
    top_features = dfs_dict[base_year].nlargest(top_n, 'mean_abs_shap')['feature'].values
    
    for i, year in enumerate(years):
        df = dfs_dict[year]
        df_filtered = df[df['feature'].isin(top_features)].set_index('feature').loc[top_features]
        
        axes[i].barh(range(len(top_features)), df_filtered['mean_abs_shap'].values)
        axes[i].set_yticks(range(len(top_features)))
        axes[i].set_yticklabels(top_features, fontsize=9)
        axes[i].set_xlabel('Mean |SHAP value|', fontsize=10)
        axes[i].set_title(f'{year}年', fontsize=12, fontweight='bold')
        axes[i].grid(axis='x', alpha=0.3)
    
    axes[0].invert_yaxis()
    plt.suptitle(f'{model_name} - 年度別上位{top_n}特徴量', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_year_comparison_bars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 棒グラフ保存: {output_dir / f'{model_name}_year_comparison_bars.png'}")
    
    
    # 2. 年度間相関ヒートマップ
    if len(years) >= 2:
        # 全ペアの相関を計算
        corr_results = []
        for i in range(len(years)):
            for j in range(i+1, len(years)):
                result = calculate_year_correlation(
                    dfs_dict[years[i]], 
                    dfs_dict[years[j]], 
                    years[i], 
                    years[j]
                )
                corr_results.append(result)
        
        # 相関行列作成
        corr_matrix = pd.DataFrame(index=years, columns=years, dtype=float)
        
        for year in years:
            corr_matrix.loc[year, year] = 1.0
        
        for result in corr_results:
            y1, y2 = result['year1'], result['year2']
            corr_matrix.loc[y1, y2] = result['spearman_r']
            corr_matrix.loc[y2, y1] = result['spearman_r']
        
        # ヒートマップ描画
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix.astype(float), 
            annot=True, 
            fmt='.3f', 
            cmap='RdYlGn', 
            vmin=-1, 
            vmax=1, 
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        plt.title(f'{model_name} - 年度間SHAP値相関 (Spearman)', fontsize=14, fontweight='bold')
        plt.xlabel('年度', fontsize=12)
        plt.ylabel('年度', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_year_correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📊 相関ヒートマップ保存: {output_dir / f'{model_name}_year_correlation_heatmap.png'}")
    
    
    # 3. 時系列トレンドプロット（上位特徴量のみ）
    if len(years) >= 3:
        # 全年度共通の特徴量
        common_features = set(dfs_dict[years[0]]['feature'])
        for year in years[1:]:
            common_features &= set(dfs_dict[year]['feature'])
        
        # 最初の年度で上位の特徴量を選択
        top_common = dfs_dict[years[0]][dfs_dict[years[0]]['feature'].isin(common_features)]\
            .nlargest(10, 'mean_abs_shap')['feature'].values
        
        # トレンドプロット
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for feature in top_common:
            values = []
            for year in years:
                df = dfs_dict[year]
                value = df[df['feature'] == feature]['mean_abs_shap'].values
                values.append(value[0] if len(value) > 0 else np.nan)
            
            ax.plot(years, values, marker='o', label=feature, linewidth=2)
        
        ax.set_xlabel('年度', fontsize=12)
        ax.set_ylabel('Mean |SHAP value|', fontsize=12)
        ax.set_title(f'{model_name} - 上位特徴量の時系列変化', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_year_trend.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📊 時系列トレンド保存: {output_dir / f'{model_name}_year_trend.png'}")


def generate_comparison_report(dfs_dict, model_name, output_dir, corr_results):
    """
    年度間比較レポート（Markdown）を生成
    
    Args:
        dfs_dict (dict): {year: DataFrame} 形式の辞書
        model_name (str): モデル名
        output_dir (Path): 出力ディレクトリ
        corr_results (list): 相関計算結果リスト
    """
    years = sorted(dfs_dict.keys())
    
    report_lines = [
        f"# {model_name} - 年度間SHAP値比較レポート\n",
        f"**分析対象年度**: {', '.join(map(str, years))}\n",
        f"**レポート生成日時**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "\n---\n\n",
        "## 1. 年度別上位特徴量\n\n"
    ]
    
    for year in years:
        df = dfs_dict[year]
        top_10 = df.nlargest(10, 'mean_abs_shap')
        
        report_lines.append(f"### {year}年\n\n")
        report_lines.append("| 順位 | 特徴量 | Mean |SHAP| |\n")
        report_lines.append("|------|--------|-------------|\n")
        
        for i, row in enumerate(top_10.itertuples(), 1):
            report_lines.append(f"| {i} | {row.feature} | {row.mean_abs_shap:.6f} |\n")
        
        report_lines.append("\n")
    
    
    # 相関統計
    if corr_results:
        report_lines.append("\n## 2. 年度間相関統計\n\n")
        report_lines.append("| 年度1 | 年度2 | 共通特徴量数 | Pearson r | Spearman ρ |\n")
        report_lines.append("|-------|-------|--------------|-----------|------------|\n")
        
        for result in corr_results:
            report_lines.append(
                f"| {result['year1']} | {result['year2']} | "
                f"{result['common_features']} | "
                f"{result['pearson_r']:.4f} | "
                f"{result['spearman_r']:.4f} |\n"
            )
        
        report_lines.append("\n")
    
    
    # トレンド分析（上昇/下降）
    if len(years) >= 3:
        report_lines.append("\n## 3. トレンド分析\n\n")
        
        # 共通特徴量
        common_features = set(dfs_dict[years[0]]['feature'])
        for year in years[1:]:
            common_features &= set(dfs_dict[year]['feature'])
        
        trend_data = []
        for feature in common_features:
            values = []
            for year in years:
                df = dfs_dict[year]
                value = df[df['feature'] == feature]['mean_abs_shap'].values
                values.append(value[0] if len(value) > 0 else np.nan)
            
            # 線形トレンド計算
            if not any(np.isnan(values)):
                slope = np.polyfit(range(len(years)), values, 1)[0]
                trend_data.append({
                    'feature': feature,
                    'slope': slope,
                    'start_value': values[0],
                    'end_value': values[-1],
                    'change_pct': ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
                })
        
        # 上昇トレンド Top 5
        trend_df = pd.DataFrame(trend_data).sort_values('slope', ascending=False)
        
        report_lines.append("### 重要度上昇トレンド Top 5\n\n")
        report_lines.append("| 特徴量 | 変化率 | 開始値 | 終了値 |\n")
        report_lines.append("|--------|--------|--------|--------|\n")
        
        for row in trend_df.head(5).itertuples():
            report_lines.append(
                f"| {row.feature} | {row.change_pct:+.2f}% | "
                f"{row.start_value:.6f} | {row.end_value:.6f} |\n"
            )
        
        report_lines.append("\n")
        
        # 下降トレンド Top 5
        report_lines.append("### 重要度下降トレンド Top 5\n\n")
        report_lines.append("| 特徴量 | 変化率 | 開始値 | 終了値 |\n")
        report_lines.append("|--------|--------|--------|--------|\n")
        
        for row in trend_df.tail(5).itertuples():
            report_lines.append(
                f"| {row.feature} | {row.change_pct:+.2f}% | "
                f"{row.start_value:.6f} | {row.end_value:.6f} |\n"
            )
        
        report_lines.append("\n")
    
    
    # ファイル保存
    report_path = output_dir / f'{model_name}_year_comparison_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    print(f"   📄 比較レポート保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='年度間SHAP値比較分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='モデル名（例: tokyo_turf_3ageup_long）'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        required=True,
        help='比較対象年度（スペース区切り、例: 2021 2022 2023）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='出力ディレクトリ（デフォルト: shap_analysis/{model}/year_comparison）'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='表示する上位特徴量数（デフォルト: 20）'
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリ設定
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path('shap_analysis') / args.model / 'year_comparison'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"📊 年度間SHAP値比較分析")
    print(f"{'='*60}")
    print(f"モデル: {args.model}")
    print(f"対象年度: {', '.join(map(str, args.years))}")
    print(f"出力先: {output_dir}")
    print(f"{'='*60}\n")
    
    # 各年度のSHAP CSVを読み込み
    dfs_dict = {}
    
    for year in args.years:
        df = load_shap_csv(args.model, year)
        if df is not None:
            dfs_dict[year] = df
    
    if len(dfs_dict) < 2:
        print("\n❌ エラー: 比較には最低2年度のデータが必要です")
        sys.exit(1)
    
    print(f"\n✅ データ読み込み完了: {len(dfs_dict)}/{len(args.years)} 年度\n")
    
    # 年度間相関計算
    years = sorted(dfs_dict.keys())
    corr_results = []
    
    print("🔍 年度間相関を計算中...\n")
    for i in range(len(years)):
        for j in range(i+1, len(years)):
            result = calculate_year_correlation(
                dfs_dict[years[i]], 
                dfs_dict[years[j]], 
                years[i], 
                years[j]
            )
            corr_results.append(result)
            print(f"   {years[i]} vs {years[j]}: Spearman ρ = {result['spearman_r']:.4f}")
    
    print()
    
    # プロット生成
    print("📊 プロット生成中...\n")
    plot_year_comparison(dfs_dict, args.model, output_dir, top_n=args.top_n)
    
    # レポート生成
    print("\n📄 レポート生成中...\n")
    generate_comparison_report(dfs_dict, args.model, output_dir, corr_results)
    
    print(f"\n{'='*60}")
    print(f"✅ 年度間比較分析完了!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
