# RAG 质量包 · 过程报告

> 所属阶段：路线排序定稿第 2 站（RAG 质量包）｜完成日期：2026-09-25｜单测：162 passed

---

## 一、这阶段做了什么

1. **混合检索**（`app/rag/hybrid.py` + retriever 分支）：BM25 关键词 + 向量语义双路召回，**双闸准入**（向量闸=`SCORE_THRESHOLD`、BM25 闸=`BM25_FLOOR`，并集放行）→ **RRF 融合排序**（k=60，只看名次不看分数量纲）；`RETRIEVAL_MODE=vector|hybrid` 配置开关，默认 vector 保基线，答辩可 A/B 演示；
2. **语料与分数口径**：`ChromaStore.get_all()` 全量拉块做 BM25 语料（**零表结构变更**）；BM25 独家命中用 `get_embeddings()` 取存量向量现算余弦——sources 的「相似度」对两类命中同口径；
3. **范围过滤复用**：页码/文件过滤谓词 `_in_scope` 两路共用，scope 能力不因混合模式失效；
4. **评估与标定脚本**（`scripts/eval_rag.py` + `config/eval_dataset.json` 模板）：走真实 HTTP 全链路，产出逐题结果表 + 分数分布 + 阈值建议，报告写入 `docs/reports/rag-eval-<label>.md`；
5. **配置**：`RETRIEVAL_MODE`、`BM25_FLOOR` 进 config 与 `.env.example`；新依赖 `jieba` + `rank_bm25`（纯 Python，经产品经理 §4 确认）。

---

## 二、遇到了什么问题

1. **★BM25 小语料 IDF=0（单测实测抓出的真 bug）**：rank_bm25 的 Okapi IDF 公式在「2 篇文档、术语出现在 1 篇」时算出 `ln(1.5/1.5)=0`——术语明明在文档里，BM25 却得 0 分，核心的「关键词捞回」用例直接空手；
2. **首跑评估：3 个正例只中 2 个**——「梯度下降学习率」问题 hit=False；
3. 评估模板负例全部 0.3s 秒回兜底，**看不到负例的擦边分数**（兜底响应不含 sources）；
4. hybrid 分支第一版语料构建写了「先收集再推倒重来」的冗余代码；
5. 测试夹具手滑：死三目表达式 + 解包顺序反了（`_, system = build...` 取到的是 prompt）。

## 三、怎么解决的问题（与上一一对应）

1. **改用恒正的平滑 IDF**：`idf=ln(1+(N-df+0.5)/(df+0.5))`——小语料恒正、大语料逼近标准公式（N=1000,df=1 时两者均 ≈6.5），只覆盖 `index.idf` 字典、BM25Okapi 的 tf 归一化机制原样不动；`test_hybrid_rescues_keyword_hit_vector_rejects` 锁死该行为（向量闸设 1.1 物理全灭 → BM25 必须救回术语块，对照组纯向量模式必须空手）；
2. 首跑分析结论：该问题**走了完整链路**（10.1s=检索命中→分类→生成→模型判「资料不足」→兜底）——说明其课件本就没有梯度下降内容（那是 demo_e2e 临时文档的知识点），**资料不足闸门按设计工作**，属模板问题集与课件不匹配，非系统缺陷；正例「数据结构三要素」「算法五特性」命中，最低分 **0.76**；
3. 擦边分数不可见是契约使然（hit=false 不带 sources）——解决路径写入验收指引：把负例改成**贴课件但说错**的问题（如「数据结构的两要素是什么」）会真实擦边，命中后分数即可见，据此定 `SCORE_THRESHOLD`；当前证据（正例 0.76 起步）支持阈值从 0.35 上调至 ~0.5 留出安全边际，待用户拍板；
4. 简化为一次成池：`texts` 字典（向量候选先入 + get_all 补缺）即全语料，键序插入序稳定、融合结果可复现；
5. 逐处修正；教训：**测试里的花活（三目占位、无序解包）比正经代码更危险**——写完必须让断言先跑起来。

**遗留**（下一步验收项）：① 用户改写 `config/eval_dataset.json` 贴课件后重跑标定，回填 `SCORE_THRESHOLD`；② `RETRIEVAL_MODE=hybrid` 重启服务做 A/B 对比（`BM25_FLOOR` 初值 2.0 待 hybrid 体验后微调）；③ rerank 明确不做（依赖过重），已属 README 升级路径叙事。

---

## 阶段内增强补记（2026-09-25 hybrid 切换后线上排障）

### 又遇到什么问题

6. **★生产 500：`ValueError: truth value of an array is ambiguous`**——用户切 `RETRIEVAL_MODE=hybrid` 后，「第一章讲了什么」类范围提问打 `/chat/ask` 必炸，普通提问正常；
7. **★越权隐患被既有测试当场抓住**：`test_group_isolation` 失败——hybrid 按 `store_map` 全键遍历而不是按 `group_ids` 授权遍历，stores 比授权宽时会捞到未授权分组的内容；
8. **测试随开发机 `.env` 漂移**：`.env` 开 hybrid 后全量 pytest 悄悄跑进混合路径，向量用例语义变了（17 连环失败的放大器）；
9. 补签名时第一版漏传 `group_ids` 参数 → NameError 连环 17 个失败。

### 又怎么解决的

6. 定位手法：**双路复现**——进程内直连检索正常（排除检索层）→ TestClient 打全链路（`raise_server_exceptions` 把服务端栈原样抛出）→ 栈钉死 `vector_store.get_embeddings` 的 `res.get("embeddings") or []`。根因：chroma 对 `include=["embeddings"]` 返回 **numpy 二维数组**，ndarray 真值判断对多元素数组直接抛错（ids/documents 是 list 才能用 `or []`）。修复为显式判 `None` 后 `list()` 逐元素转 float；新增 `test_get_embeddings_roundtrip_handles_numpy` 回归（该路径只有「BM25 独家命中补余弦」会走，普通提问测不到——所以必须单独立测）。修复后四种场景（普通/章节/出题/页码）全链路 200 实测通过；
7. hybrid 循环改为 **`group_ids ∩ store_map` 交集**（`group_ids` 是授权权威，与向量路径同口径），注释写明被哪个测试抓住、为什么这是越权而非 bug 修辞；
8. conftest 加 autouse 夹具 `_pin_retrieval_mode_vector`：默认钉死 vector 保证基线可复现，测 hybrid 的用例体内自行覆盖（测试内 setattr 晚于夹具必然生效）——测试结果从此不随开发机 `.env` 漂移；
9. 补齐 `_retrieve_hybrid(group_ids=...)` 签名与调用点，全量 163 passed。

---

## 标定执行补记（2026-09-25，双模式实测）

- 用自有实例（8001 端口，`RETRIEVAL_MODE` 分别覆写 vector/hybrid）跑完两轮 9 题评估，报告：`rag-eval-vector.md` / `rag-eval-hybrid.md`；
- **脚本自动建议 0.70 被人工复核推翻**：三条「擦边负例」（二叉树/图/哈希）实际被《思维导图》全课程 PDF 与试卷解析覆盖，命中带真来源=系统正确——**负例设计必须先查课件覆盖面**，此为本阶段最大方法论教训；
- 有效数据：正例最低 0.655、远负例 0.35 下全空 → **回填 `.env`：`SCORE_THRESHOLD=0.5`**（留 0.15 正例余量、比原值严 0.15）；
- BM25 词面量化的反直觉发现：「番茄炒蛋」15.35 分 > 「线性表」7.14 分（共用高频词堆叠）——`BM25_FLOOR` 无法分离 topic 垃圾，保持默认 2.0；兜底正确性两轮 100%，双闸（向量阈值+资料不足）实测有效；
- A/B 结论如实记录于 hybrid 报告：本问题集上 hybrid 无准确率增益、远负例多耗 ~3s，开关保留作答辩演示与术语场景预案。
