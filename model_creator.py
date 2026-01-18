import psycopg2
import os
from pathlib2 import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import optuna
from sklearn.metrics import ndcg_score
from keiba_constants import get_track_name, format_model_description
from datetime import datetime
from db_query_builder import build_race_data_query


def create_universal_model(track_code, kyoso_shubetsu_code, surface_type, 
                          min_distance, max_distance, model_filename, output_dir='models',
                          year_start=2013, year_end=2022):
    """
    汎用的な競馬予測モデル作成関数
    
    Args:
        track_code (str): 競馬場コード ('01'=札幌, '02'=函館, ..., '09'=阪神, '10'=小倉)
        kyoso_shubetsu_code (str): 競争種別コード ('12'=3歳, '13'=3歳以上)
        surface_type (str): 'turf' or 'dirt' (芝またはダート)
        min_distance (int): 最小距離 (例: 1000)
        max_distance (int): 最大距離 (例: 3600, 上限なしの場合は9999)
        model_filename (str): 保存するモデルファイル名 (例: 'hanshin_turf_3ageup.sav')
        output_dir (str): モデル保存先ディレクトリ (デフォルト: 'models')
        year_start (int): 学習データ開始年 (デフォルト: 2013)
        year_end (int): 学習データ終了年 (デフォルト: 2022)
    
    Returns:
        None: モデルファイルを保存
    """
    
    # カレントディレクトリをスクリプト配置箇所に変更
    os.chdir(Path(__file__).parent)
    print(f"作業ディレクトリ:{os.getcwd()}")
    
    # モデル保存用ディレクトリを作成（存在しない場合）
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"[FILE] モデル保存先: {output_path.absolute()}")

    # PostgreSQL コネクションの作成
    conn = psycopg2.connect(
        host='localhost',
        port='5432',
        user='postgres',
        password='ahtaht88',
        dbname='keiba'
    )

    # SQLクエリを共通化モジュールで生成
    sql = build_race_data_query(
        track_code=track_code,
        year_start=year_start,
        year_end=year_end,
        surface_type=surface_type,
        distance_min=min_distance,
        distance_max=max_distance,
        kyoso_shubetsu_code=kyoso_shubetsu_code,
        include_payout=False  # model_creator.pyでは払い戻し情報不要
    )

    # モデル説明を生成
    model_desc = format_model_description(track_code, kyoso_shubetsu_code, surface_type, min_distance, max_distance)
    print(f"[RACE] モデル作成開始: {model_desc}")
    
    # SQLをログファイルに出力（常に上書き）
    log_filepath = Path('sql_log.txt')
    with open(log_filepath, 'w', encoding='utf-8') as f:
        f.write(f"=== モデル作成SQL ===\n")
        f.write(f"モデル: {model_desc}\n")
        f.write(f"作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n{sql}\n")
    print(f"[NOTE] SQLをログファイルに出力: {log_filepath}")
    
    # SELECT結果をDataFrame
    df = pd.read_sql_query(sql=sql, con=conn)
    
    if len(df) == 0:
        print("[ERROR] 指定した条件に合致するデータが見つかりませんでした。条件を確認してください。")
        return

    print(f"[+] データ件数: {len(df)}件")

    # 着順スコアが0のデータは無効扱いにして除外
    df = df[df['chakujun_score'] > 0]

    # データ前処理（共通化モジュール使用）
    from data_preprocessing import preprocess_race_data
    df = preprocess_race_data(df, verbose=True)

    # 特徴量作成（共通化モジュール使用）
    from feature_engineering import create_features, add_advanced_features
    
    # 基本特徴量を作成
    X = create_features(df)
    
    # 高度な特徴量を追加（feature_engineering.pyで共通化）
    print("[START] 高度な特徴量生成...")
    X = add_advanced_features(
        df=df,
        X=X,
        surface_type=surface_type,
        min_distance=min_distance,
        max_distance=max_distance,
        logger=None,
        inverse_rank=True  # model_creator.pyでは着順を反転
    )
    print(f"[OK] 特徴量生成完了: {len(X.columns)}個")

    categorical_features = []

    # 距離別特徴量選択はadd_advanced_features()内で実施済み
    print(f"\n[INFO] 特徴量リスト: {list(X.columns)}")

    #目的変数を設定
    y = df['kakutei_chakujun_numeric'].astype(int)

    # レースごとのグループを正しく計算
    df['group_id'] = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango']).ngroup()
    groups = df['group_id'].values
    print(f"グループ数: {len(set(groups))}")  # ユニークなグループの数
    print(f"データ数: {len(groups)}")  # 全データ数

    # データとグループ数が一致していることを確認
    if len(groups) != len(X):
        raise ValueError(f"データ件数({len(X)})とグループの数({len(groups)})が一致しません！")

    # 改善1: 時系列分割を年単位で明確化
    # 年単位で訓練/テストを分割（ばらつき削減のため）
    # 例: 2013-2020年を訓練、2021-2022年をテスト
    
    # 学習データの年範囲から訓練/テスト年を計算
    all_years = sorted(df['kaisai_nen'].unique())
    total_years = len(all_years)
    
    # 75%を訓練データに（年単位で）
    train_year_count = int(total_years * 0.75)
    train_years = all_years[:train_year_count]
    test_years = all_years[train_year_count:]
    
    print(f"[DATE] 訓練データ年: {train_years} ({len(train_years)}年間)")
    print(f"[DATE] テストデータ年: {test_years} ({len(test_years)}年間)")
    
    # 年単位でインデックスを分割
    train_indices = df[df['kaisai_nen'].isin(train_years)].index
    test_indices = df[df['kaisai_nen'].isin(test_years)].index
    
    # データ分割
    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]
    groups_train = groups[train_indices]
    groups_test = groups[test_indices]
    
    print(f"[OK] 訓練データ件数: {len(X_train)}件")
    print(f"[OK] テストデータ件数: {len(X_test)}件")

    # Optunaのobjective関数
    def objective(trial):
        param = {
            'objective': 'lambdarank',
            'metric': 'ndcg',                                                              # 上位の並び順を重視
            'ndcg_eval_at': [1, 3, 5],
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_uniform('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 10.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 5)
        }

        # 修正: グループサイズを正しい順序で計算（データの順序を保持）
        # sort=Falseで元のデータ順を維持しながらグループサイズを抽出
        train_df_with_group = pd.DataFrame({'group': groups_train}).reset_index(drop=True)
        train_group_sizes = train_df_with_group.groupby('group', sort=False).size().values
        
        test_df_with_group = pd.DataFrame({'group': groups_test}).reset_index(drop=True)
        test_group_sizes = test_df_with_group.groupby('group', sort=False).size().values
        
        dtrain = lgb.Dataset(X_train, label=y_train, group=train_group_sizes, categorical_feature=categorical_features)
        dvalid = lgb.Dataset(X_test, label=y_test, group=test_group_sizes, categorical_feature=categorical_features)

        tmp_model = lgb.train(
            param,
            dtrain,
            valid_sets=[dvalid],
            valid_names=['valid'],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # 修正: レースごとにNDCGを計算して平均を返す（正しい評価方法）
        preds = tmp_model.predict(X_test, num_iteration=tmp_model.best_iteration)
        
        # テストデータをレースごとに分割してNDCGを計算
        ndcg_scores = []
        start_idx = 0
        for group_size in test_group_sizes:
            end_idx = start_idx + group_size
            y_true_group = y_test.iloc[start_idx:end_idx].values
            y_pred_group = preds[start_idx:end_idx]
            
            # レース内に2頭以上いる場合のみNDCG計算（1頭だとエラーになる）
            if len(y_true_group) > 1:
                # 2次元配列として渡す
                ndcg = ndcg_score([y_true_group], [y_pred_group], k=5)
                ndcg_scores.append(ndcg)
            
            start_idx = end_idx
        
        # 全レースのNDCG平均を返す
        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    # [TIP]推奨: 本番運用前に上位パラメータを複数シード（random_state=42,43,44...）で
    #        再学習し、NDCG/ROI/的中率の安定性を確認することを強く推奨
    # TODO 将来の改善: lgb.Dataset の weight パラメータにオッズベースの重みを導入し、
    #      穴馬（高オッズ馬）の予測精度を向上させることでROI最適化を図る
    
    # 改善2: Optunaのシード固定（再現性向上のため）
    print("[TEST] ハイパーパラメータ最適化を開始...")
    sampler = optuna.samplers.TPESampler(seed=42)  # シードを固定
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=50)

    print('Best trial:')
    print(study.best_trial.params)

    # 最適パラメータを使って再学習
    best_params = study.best_trial.params

    best_params.update({
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'boosting_type': 'gbdt',
        'verbosity': 0,  # 学習の進捗を表示
        'random_state': 42,
    })

    # 修正: グループデータを正しく準備（データの順序を保持）
    # レースごとの出走頭数を計算（sort=Falseで元の順序を維持）
    train_df_with_group = pd.DataFrame({'group': groups_train}).reset_index(drop=True)
    train_group_sizes = train_df_with_group.groupby('group', sort=False).size().values
    
    test_df_with_group = pd.DataFrame({'group': groups_test}).reset_index(drop=True)
    test_group_sizes = test_df_with_group.groupby('group', sort=False).size().values
    
    print(f"訓練データのレース数: {len(train_group_sizes)}")
    print(f"テストデータのレース数: {len(test_group_sizes)}")
    
    # LightGBM用のデータセットを作成
    dtrain = lgb.Dataset(X_train, label=y_train, group=train_group_sizes, categorical_feature=categorical_features)
    dvalid = lgb.Dataset(X_test, label=y_test, group=test_group_sizes, categorical_feature=categorical_features)

    # 最適化されたパラメータでモデルを学習
    print(" 最適化されたパラメータでモデルを学習するよ！")
    model = lgb.train(
        best_params,
        dtrain,
        valid_sets=[dvalid],
        valid_names=['テストデータ'],
        num_boost_round=1000,  # 最大反復回数
        callbacks=[
            lgb.early_stopping(50),  # 50回改善がなければ早期終了（学習の安定化）
            lgb.log_evaluation(10)   # 10回ごとに結果表示
        ]
    )

    # 特徴量の重要度を確認する（モデルがどの情報を重視しているか）
    importance = model.feature_importance()
    feature_names = X.columns
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("特徴量の重要度:")
    print(feature_importance)

    # モデルを保存する
    model_filepath = output_path / model_filename
    pickle.dump(model, open(model_filepath, 'wb'))
    print(f"[OK] モデルを {model_filepath} に保存しました")

    # コネクションをクローズ
    conn.close()


# 旧来の関数を維持（互換性のため）
def make_model():
    """
    旧バージョン互換性のための関数
    阪神競馬場の３歳以上芝中長距離モデルを作成
    
    注意: 互換性のため、ルートディレクトリに保存されます
    """
    create_universal_model(
        track_code='09',  # 阪神
        kyoso_shubetsu_code='13',  # 3歳以上
        surface_type='turf',  # 芝
        min_distance=1700,  # 中長距離
        max_distance=9999,  # 上限なし
        model_filename='hanshin_shiba_3ageup_model.sav',
        output_dir='.'  # ルートディレクトリに保存（既存の動作を維持）
    )


if __name__ == '__main__':
    # テスト実行：旧バージョンと同じモデルを作成
    make_model()