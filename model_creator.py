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


def create_universal_model(track_code, kyoso_shubetsu_code, surface_type, 
                          min_distance, max_distance, model_filename, output_dir='models'):
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
    
    Returns:
        None: モデルファイルを保存
    """
    
    # カレントディレクトリをスクリプト配置箇所に変更
    os.chdir(Path(__file__).parent)
    print(f"作業ディレクトリ:{os.getcwd()}")
    
    # モデル保存用ディレクトリを作成（存在しない場合）
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"📁 モデル保存先: {output_path.absolute()}")

    # PostgreSQL コネクションの作成
    conn = psycopg2.connect(
        host='localhost',
        port='5432',
        user='postgres',
        password='ahtaht88',
        dbname='keiba'
    )

    # トラック条件を動的に設定
    if surface_type.lower() == 'turf':
        # 芝の場合
        # TODO 芝は10～22と広く範囲指定する。芝とダートで精度に違いが出るようであれば対象トラックコードを減らすなどの工夫が必要かも。
        track_condition = "cast(rase.track_code as integer) between 10 and 22"
        baba_condition = "ra.babajotai_code_shiba"
    else:
        # ダートの場合
        track_condition = "cast(rase.track_code as integer) between 23 and 24"
        baba_condition = "ra.babajotai_code_dirt"

    # 距離条件を設定
    if max_distance == 9999:
        distance_condition = f"cast(rase.kyori as integer) >= {min_distance}"
    else:
        distance_condition = f"cast(rase.kyori as integer) between {min_distance} and {max_distance}"
    
    # 競争種別を設定
    if kyoso_shubetsu_code == '12':
        # 3歳戦
        kyoso_shubetsu_condition = "cast(rase.kyoso_shubetsu_code as integer) = 12"
    elif kyoso_shubetsu_code == '13':
        kyoso_shubetsu_condition = "cast(rase.kyoso_shubetsu_code as integer) >= 13"

    # SQLクエリを動的に生成
    sql = f"""
    select * from (
        select
        ra.kaisai_nen,
        ra.kaisai_tsukihi,
        ra.keibajo_code,
        CASE 
            WHEN ra.keibajo_code = '01' THEN '札幌' 
            WHEN ra.keibajo_code = '02' THEN '函館' 
            WHEN ra.keibajo_code = '03' THEN '福島' 
            WHEN ra.keibajo_code = '04' THEN '新潟' 
            WHEN ra.keibajo_code = '05' THEN '東京' 
            WHEN ra.keibajo_code = '06' THEN '中山' 
            WHEN ra.keibajo_code = '07' THEN '中京' 
            WHEN ra.keibajo_code = '08' THEN '京都' 
            WHEN ra.keibajo_code = '09' THEN '阪神' 
            WHEN ra.keibajo_code = '10' THEN '小倉' 
            ELSE '' 
        END keibajo_name,
        ra.race_bango,
        ra.kyori,
        ra.tenko_code,
        {baba_condition} as babajotai_code,
        ra.grade_code,
        ra.kyoso_joken_code,
        ra.kyoso_shubetsu_code,
        ra.track_code,
        seum.ketto_toroku_bango,
        trim(seum.bamei),
        seum.wakuban,
        cast(seum.umaban as integer) as umaban_numeric,
        seum.barei,
        seum.kishu_code,
        seum.chokyoshi_code,
        seum.futan_juryo,
        seum.seibetsu_code,
        CASE WHEN seum.seibetsu_code = '1' THEN '1' ELSE '0' END AS mare_horse,
        CASE WHEN seum.seibetsu_code = '2' THEN '1' ELSE '0' END AS femare_horse,
        CASE WHEN seum.seibetsu_code = '3' THEN '1' ELSE '0' END AS sen_horse,
        CASE WHEN {baba_condition} = '1' THEN '1' ELSE '0' END AS baba_good,
        CASE WHEN {baba_condition} = '2' THEN '1' ELSE '0' END AS baba_slightly_heavy,
        CASE WHEN {baba_condition} = '3' THEN '1' ELSE '0' END AS baba_heavy,
        CASE WHEN {baba_condition} = '4' THEN '1' ELSE '0' END AS baba_defective,
        CASE WHEN ra.tenko_code = '1' THEN '1' ELSE '0' END AS tenko_fine,
        CASE WHEN ra.tenko_code = '2' THEN '1' ELSE '0' END AS tenko_cloudy,
        CASE WHEN ra.tenko_code = '3' THEN '1' ELSE '0' END AS tenko_rainy,
        CASE WHEN ra.tenko_code = '4' THEN '1' ELSE '0' END AS tenko_drizzle,
        CASE WHEN ra.tenko_code = '5' THEN '1' ELSE '0' END AS tenko_snow,
        CASE WHEN ra.tenko_code = '6' THEN '1' ELSE '0' END AS tenko_light_snow,
        18 - cast(seum.kakutei_chakujun as integer) + 1 as kakutei_chakujun_numeric, 
        1.0 / nullif(cast(seum.kakutei_chakujun as integer), 0) as chakujun_score,  --上位着順ほど1に近くなる
        1 - (cast(seum.kakutei_chakujun as float) / cast(ra.shusso_tosu as float)) as sotai_chakujun_numeric,
        cast(ra.kyori as integer) /
        (
        FLOOR(cast(seum.soha_time as integer) / 1000) * 60 +
        FLOOR((cast(seum.soha_time as integer) % 1000) / 10) +
        (cast(seum.soha_time as integer) % 10) * 0.1
        ) AS time_index,
        SUM(
            CASE 
                WHEN seum.kakutei_chakujun = '01' THEN 100
                WHEN seum.kakutei_chakujun = '02' THEN 80
                WHEN seum.kakutei_chakujun = '03' THEN 60
                WHEN seum.kakutei_chakujun = '04' THEN 40
                WHEN seum.kakutei_chakujun = '05' THEN 30
                WHEN seum.kakutei_chakujun = '06' THEN 20
                WHEN seum.kakutei_chakujun = '07' THEN 10
                ELSE 5 
            END
            * CASE 
                WHEN ra.grade_code = 'A' THEN 1.00                                                                                          --G1
                WHEN ra.grade_code = 'B' THEN 0.80                                                                                          --G2
                WHEN ra.grade_code = 'C' THEN 0.60                                                                                          --G3
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '999' THEN 0.50       --OP
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '016' THEN 0.40       --3勝クラス
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '010' THEN 0.30       --2勝クラス
                WHEN ra.grade_code <> 'A' AND ra.grade_code <> 'B' AND ra.grade_code <> 'C' AND ra.kyoso_joken_code = '005' THEN 0.20       --1勝クラス
                ELSE 0.10                                                                                                                   --未勝利
            END
        ) OVER (
            PARTITION BY seum.ketto_toroku_bango
            ORDER BY cast(ra.kaisai_nen as integer), cast(ra.kaisai_tsukihi as integer)
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING  
        ) AS past_score  --グレード別スコア
        ,cast(seum.kohan_3f AS FLOAT) / 10 as kohan_3f_sec
        ,CASE 
            WHEN cast(seum.kohan_3f as integer) > 0 THEN
            -- 標準タイムからの差に変換（小さいほど速い）
            CAST(seum.kohan_3f AS FLOAT) / 10 - 
            -- 距離ごとの基準タイム (距離に応じた補正)
            CASE
                WHEN cast(ra.kyori as integer) <= 1600 THEN 33.5  -- マイル以下
                WHEN cast(ra.kyori as integer) <= 2000 THEN 35.0  -- 中距離
                WHEN cast(ra.kyori as integer) <= 2400 THEN 36.0  -- 中長距離
                ELSE 37.0  -- 長距離
            END
            ELSE 0
        END AS kohan_3f_index
    from
        jvd_ra ra 
        inner join ( 
            select
                se.kaisai_nen
                , se.kaisai_tsukihi
                , se.keibajo_code
                , se.race_bango
                , se.kakutei_chakujun
                , se.ketto_toroku_bango
                , se.bamei
                , se.wakuban
                , se.umaban
                , se.barei
                , se.seibetsu_code
                , se.kishu_code
                , se.chokyoshi_code
                , se.futan_juryo
                , se.kohan_3f
                , se.soha_time
            from
                jvd_se se
            where 
                se.kohan_3f <> '000' 
                and se.kohan_3f <> '999'
        ) seum 
            on ra.kaisai_nen = seum.kaisai_nen 
            and ra.kaisai_tsukihi = seum.kaisai_tsukihi 
            and ra.keibajo_code = seum.keibajo_code 
            and ra.race_bango = seum.race_bango 
    where
        cast(ra.kaisai_nen as integer) between 2013 and 2022                  --2013～2022
    ) rase 
    where 
    rase.keibajo_code = '{track_code}'                                        --競馬場指定
    and {kyoso_shubetsu_condition}                                            --競争種別
    and {track_condition}                                                     --芝/ダート
    and {distance_condition}                                                  --距離条件
    """

    # モデル説明を生成
    model_desc = format_model_description(track_code, kyoso_shubetsu_code, surface_type, min_distance, max_distance)
    print(f"🏇 モデル作成開始: {model_desc}")
    
    # SELECT結果をDataFrame
    df = pd.read_sql_query(sql=sql, con=conn)
    
    if len(df) == 0:
        print("❌ 指定した条件に合致するデータが見つかりませんでした。条件を確認してください。")
        return

    print(f"📊 データ件数: {len(df)}件")

    # 着順スコアが0のデータは無効扱いにして除外
    df = df[df['chakujun_score'] > 0]

    # まずデータの前処理をしっかり行う
    df = df.apply(pd.to_numeric, errors='coerce')  # 数値に変換
    df = df.replace('0', np.nan)  # 0をNaNに置換
    df = df.fillna(0)  # 欠損値を0に置換

    X = df.loc[:, [
        "kyori",
        "tenko_code",  
        "babajotai_code",
        "seibetsu_code",  
        # "umaban_numeric", 
        # "barei",  
        "futan_juryo",
        "past_score",
        "kohan_3f_index",
        "sotai_chakujun_numeric",
        "time_index",
    ]].astype(float)
    
    # 高性能な派生特徴量を追加！
    # 枠番と頭数の比率（内枠有利度）
    max_wakuban = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['wakuban'].transform('max')
    df['wakuban_ratio'] = df['wakuban'] / max_wakuban
    X['wakuban_ratio'] = df['wakuban_ratio']
    
    # 斤量と馬齢の比率（若馬の負担能力）
    df['futan_per_barei'] = df['futan_juryo'] / df['barei'].replace(0, 1)
    X['futan_per_barei'] = df['futan_per_barei']

    # 🚀 高精度二次特徴量を追加（予測スコア重複回避 + 精度向上）
    # シンプルな特徴量から始めて過学習を防ぐ
    
    # 馬番×距離の相互作用（内外枠の距離適性）
    df['umaban_kyori_interaction'] = df['umaban_numeric'] * df['kyori'] / 1000  # スケール調整
    X['umaban_kyori_interaction'] = df['umaban_kyori_interaction']
    
    # 馬齢の非線形変換（競走馬のピーク年齢効果）
    # df['barei_squared'] = df['barei'] ** 2
    # X['barei_squared'] = df['barei_squared']
    df['barei_peak_distance'] = abs(df['barei'] - 4)  # 4歳をピークと仮定
    X['barei_peak_distance'] = df['barei_peak_distance']

    # レース内での馬番相対位置（頭数による正規化）
    df['umaban_percentile'] = df.groupby(['kaisai_nen', 'kaisai_tsukihi', 'race_bango'])['umaban_numeric'].transform(
        lambda x: x.rank(pct=True)
    )
    X['umaban_percentile'] = df['umaban_percentile']
    
    # # 微小な個体識別子を追加（重複完全回避のため）
    # # 馬番ベースの極小調整値
    # df['micro_adjustment'] = df['umaban_numeric'] / 1000000  # 0.000001〜0.000018程度
    # X['micro_adjustment'] = df['micro_adjustment']

    # カテゴリ変数を作成
    X['kyori'] = X['kyori'].astype('category')
    X['tenko_code'] = X['tenko_code'].astype('category')
    X['babajotai_code'] = X['babajotai_code'].astype('category')
    X['seibetsu_code'] = X['seibetsu_code'].astype('category')
    categorical_features = ['kyori', 'tenko_code', 'babajotai_code', 'seibetsu_code']

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

    # 🔥ここから変更🔥
    # 時系列でデータを分割（古い年月を訓練データに、新しい年月をテストデータに）
    # まず日付でソート
    df['kaisai_date'] = df['kaisai_nen'].astype(str) + df['kaisai_tsukihi'].astype(str).str.zfill(4)
    sorted_df = df.sort_values('kaisai_date')
    
    # データの75%を訓練データ、25%をテストデータに
    train_size = int(len(sorted_df) * 0.75)
    
    # 時系列順にインデックスを分ける
    train_indices = sorted_df.index[:train_size]
    test_indices = sorted_df.index[train_size:]
    
    # 分割したインデックスを使ってデータ分割
    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]
    groups_train = groups[train_indices]
    groups_test = groups[test_indices]
    
    # 確認してみる
    train_dates = sorted_df.loc[train_indices, 'kaisai_date'].unique()
    test_dates = sorted_df.loc[test_indices, 'kaisai_date'].unique()
    print(f"訓練データの日付範囲: {min(train_dates)} 〜 {max(train_dates)}")
    print(f"テストデータの日付範囲: {min(test_dates)} 〜 {max(test_dates)}")

    # Optunaのobjective関数
    def objective(trial):
        param = {
            'objective': 'lambdarank',
            'metric': 'ndcg',                                                              # 上位5着までの並び順を重視
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        }

        # グループごとの頭数に変換！←ここがポイント！
        train_group_sizes = pd.Series(groups_train).value_counts().sort_index().values
        test_group_sizes = pd.Series(groups_test).value_counts().sort_index().values
        
        dtrain = lgb.Dataset(X_train, label=y_train, group=train_group_sizes, categorical_feature=categorical_features)
        dvalid = lgb.Dataset(X_test, label=y_test, group=test_group_sizes, categorical_feature=categorical_features)

        tmp_model = lgb.train(
            param,
            dtrain,
            valid_sets=[dvalid],
            valid_names=['valid'],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )

        preds = tmp_model.predict(X_test, num_iteration=tmp_model.best_iteration)
        ndcg = ndcg_score([y_test.values], [preds], k=10)

        return ndcg

    # Optunaのスタディ作成＆最適化実行
    print("🔍 ハイパーパラメータ最適化を開始...")
    study = optuna.create_study(direction="maximize")
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

    # グループデータを正しく準備
    # レースごとの出走頭数を計算
    train_group_sizes = pd.Series(groups_train).value_counts().sort_index().values
    test_group_sizes = pd.Series(groups_test).value_counts().sort_index().values
    
    print(f"訓練データのレース数: {len(train_group_sizes)}")
    print(f"テストデータのレース数: {len(test_group_sizes)}")
    
    # LightGBM用のデータセットを作成
    dtrain = lgb.Dataset(X_train, label=y_train, group=train_group_sizes, categorical_feature=categorical_features)
    dvalid = lgb.Dataset(X_test, label=y_test, group=test_group_sizes, categorical_feature=categorical_features)

    # 最適化されたパラメータでモデルを学習
    print("🔥 最適化されたパラメータでモデルを学習するよ！")
    model = lgb.train(
        best_params,
        dtrain,
        valid_sets=[dvalid],
        valid_names=['テストデータ'],
        num_boost_round=1000,  # 最大反復回数
        callbacks=[
            lgb.early_stopping(30),  # 30回改善がなければ早期終了
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
    print(f"✅ モデルを {model_filepath} に保存しました")

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