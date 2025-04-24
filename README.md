using julia to do some sci-ml or other things
# 🌡️ One-Line Temperature Forecasting with Julia  
# 🌡️ Juliaでのワンライン気温予測

> A comparative modeling project using Julia to predict tomorrow's temperature using only one day's data. This study evaluates and contrasts differential equations, deep learning, and probabilistic methods within a unified experimental design.  
> 本プロジェクトは、**1日の気温データ**のみを使って翌日の気温を予測することを目的に、Juliaを用いて異なるアプローチ（微分方程式、ディープラーニング、ベイズ推論）を比較検証した実験的な取り組みです。

---

## 🧠 Project Background & Motivation  
## 🧠 プロジェクトの背景と動機

This project challenges conventional time-series forecasting approaches by minimizing the input: **Can we predict tomorrow’s temperature using only one line of daily data?**

Instead of relying on long sequences of past data, I investigate how different modeling paradigms handle minimal, high-noise, nonlinear input. It’s both a practical and philosophical inquiry into model expressiveness, generalization, and design across scientific and machine learning domains.

本プロジェクトでは、一般的な時系列予測手法とは異なり、入力を極限まで制限し、**「たった1行のデータで予測できるのか？」**という挑戦的なテーマに取り組んでいます。

長期間の履歴データに頼るのではなく、最小限でノイズの多い非線形な入力に対して、異なるモデリングアプローチがどのように対応できるかを比較しました。これは実践的かつモデリング思想的な探求でもあります。

---

## 📊 Dataset  
## 📊 データ概要

- **Source**: Daily temperature data from the Japan Meteorological Agency  
- **Years**: 2020, 2021, 2022  
- **Target**: Max daily temperature  
- **Format**: CSV, 365 entries/year  
- **Preprocessing**: Transformed into `Float32` matrices using `CSV.jl` & `DataFrames.jl`

- **出典**: 日本気象庁の気温観測データ  
- **対象年**: 2020年〜2022年  
- **使用項目**: 日最高気温  
- **形式**: CSV（365行）  
- **前処理**: `CSV.jl`, `DataFrames.jl` で読み込み、Float32行列に変換

---

## 🧮 Method 1: Differential Equations  
## 🧮 方法①：微分方程式によるモデリング

### 📌 Principle  
Based on heat transfer and energy conservation:

\[
\frac{dT}{dt} = f(T, t)
\]

### 📦 Tools
- `DifferentialEquations.jl`
- `DiffEqFlux.jl`

### 🧪 Techniques Tried
- Manually derived ODEs
- Neural ODEs
- Nonlinear input transformations (e.g., \(x^4\))
- Second-order ODEs
- GRU/LSTM hybrids

### ❌ Limitations
- Requires physics knowledge
- Slow and unstable
- Inaccurate in noisy real-world data

### 📌 原理  
熱エネルギー保存・放射則（ステファン＝ボルツマンの法則）に基づく：

\[
\frac{dT}{dt} = f(T, t)
\]

### 📦 使用ライブラリ
- `DifferentialEquations.jl`
- `DiffEqFlux.jl`

### 🧪 試した手法
- 手作業による物理微分方程式
- Neural ODE
- 非線形入力（例：\(x^4\)）
- 2階微分方程式
- GRU・LSTM ハイブリッド構造

### ❌ 課題
- モデル構築が難解
- 数値不安定・遅い
- 現実世界のノイズに弱い

---

## 🤖 Method 2: Machine Learning with Flux.jl  
## 🤖 方法②：Fluxによるニューラルネットワーク

### 📌 Principle  
Supervised learning approximation:  
\[
T_{\text{today}} \rightarrow T_{\text{tomorrow}}
\]

### 📦 Tools
- `Flux.jl`

### 🧪 Models
- MLP
- LSTM
- GRU
- Transformer

### ✅ Strengths
- High accuracy
- Stable training
- GRU and Transformer were top performers

### 📌 原理  
ニューラルネットワークで以下の写像を近似：

\[
T_{\text{today}} \rightarrow T_{\text{tomorrow}}
\]

### 📦 使用ライブラリ
- `Flux.jl`

### 🧪 モデル
- MLP
- LSTM
- GRU（最も精度良好）
- Transformer（最高の近似性能）

### ✅ 強み
- 予測精度が高い
- 安定した学習が可能

---

## 📈 Method 3: Probabilistic Modeling  
## 📈 方法③：確率モデルによる予測（ベイズ推論）

### 📌 Principle  
Modeling uncertainty via posterior:

\[
P(\theta | \text{Data}) = \frac{P(\text{Data}|\theta)P(\theta)}{P(\text{Data})}
\]

### 📦 Tools
- `Turing.jl`
- `AbstractGPs.jl`

### 🧪 Models
- Bayesian Neural Network (BNN)
- Gaussian Process (GP)
- Deep Kernel Learning (DKL)

### ❌ Limitations
- Very slow
- Complex and fragile
- Accuracy lower than deep learning

### 📌 原理  
不確実性を確率分布で表現：

\[
P(\theta | \text{Data}) = \frac{P(\text{Data}|\theta)P(\theta)}{P(\text{Data})}
\]

### 📦 使用ライブラリ
- `Turing.jl`
- `AbstractGPs.jl`

### 🧪 モデル
- ベイズニューラルネットワーク（BNN）
- ガウス過程（GP）
- 深層カーネル学習（DKL）

### ❌ 課題
- 非常に計算が重い
- 複雑で不安定
- 精度はNNより低い

---

## 🧾 Comparative Summary  
## 🧾 手法比較まとめ

| Method         | Performance | Pros                  | Cons                   |
|----------------|-------------|------------------------|------------------------|
| Differential Eq| ❌ Poor      | Physically grounded    | Complex, inaccurate    |
| MLP            | ⚪ Average   | Easy to implement      | Limited capacity       |
| LSTM           | ✅ Good      | Captures sequences     | Slower training        |
| GRU            | 🌟 Best      | Accurate + fast        | Needs tuning           |
| Transformer    | 🔥 Excellent | Powerful approximation | Heavy and complex      |
| Bayesian Models| ❌ Poor      | Uncertainty modeling   | Slow, unstable         |

---

## 🧪 My Original Contributions  
## 🧪 私の独自の取り組み・工夫

This project stands out because of its **cross-paradigm comparative design**. Instead of sticking to one method, I explored how **physics, deep learning, and probability theory** approach the same minimalist task.

- Tested multiple **modeling philosophies** on the same problem
- Developed **hybrid systems** (Neural ODE + GRU)
- Built a **Transformer** from scratch using Flux
- Implemented **Deep Kernel Learning** using Julia’s advanced libraries
- Documented **failures and insights**, not just successes
- Compared methods not only on accuracy, but also on **theoretical assumptions and computational cost**

本プロジェクトの最も特徴的な点は、**全く異なるモデリング分野を横断的に比較**したところにあります。

- 1行の気温データで予測するという同一問題に、物理モデル・機械学習・確率モデルを適用
- Neural ODEとGRUを組み合わせた**ハイブリッド構造**を開発
- Fluxで**Transformerを自作**
- Juliaの高度なライブラリで**Deep Kernel Learning**を試行
- 成功例だけでなく**失敗と考察も詳細に記録**
- 精度だけでなく、**理論的背景と計算コスト**も評価対象に含めた

---

## 📚 References  
## 📚 参考文献

- Flux.jl  
- Turing.jl  
- DiffEqFlux.jl  
- AbstractGPs.jl  
- Japan Meteorological Agency / 日本気象庁

---




