
# ex1 - 信用卡詐欺偵測練習

## 資料集
- 來自 Kaggle: `mlg-ulb/creditcardfraud`
- 含 284,807 筆交易資料，492 筆為詐欺（約 0.172%）

## 任務目標
1. 實作 **監督式學習**模型（Random Forest）
2. 實作 **非監督式學習**模型（KMeans）
3. 嘗試透過優化提升模型效能

## 模型與結果

### 🎯 監督式學習：Random Forest

| 模型版本           | Precision | Recall | F1 Score |
|--------------------|-----------|--------|----------|
| 原始 RF            | 0.94      | 0.82   | 0.88     |
| RF（Balanced）     | 0.97      | 0.77   | 0.86     |

> 使用 `class_weight='balanced'` 可提升對少數類別的關注，有效提升精確率。

### 🔍 非監督式學習：KMeans

| Precision | Recall | F1 Score |
|-----------|--------|----------|
| 0.78      | 0.36   | 0.50     |

> 表現雖不如 RF，但在無標籤情況下仍有不錯的 precision，可作為輔助工具。

## 結論
- Random Forest 經適當調參後能有效偵測詐欺交易。
- KMeans 可作為無監督的異常預警機制。
- 建議未來結合兩種方法（如 IsolationForest + RF）進行模型融合，可能進一步提升 recall。
