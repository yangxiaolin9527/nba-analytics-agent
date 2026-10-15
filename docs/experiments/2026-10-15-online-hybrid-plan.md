# 在线 B/C 能力分流：实施与冻结评估计划

> 使用 writing-plans / executing-plans，在现有隔离工作区 `/tmp/nba-two-stage-sql` 顺序执行；用户已批准本轮实验，不修改生产默认路线，不自动提交。

**目标**：验证真实请求中，先共享语义解析与数据库 grounding，再只执行 B 或 C，能否保留现有查询质量并扩展分析覆盖。

**已批准设计**：沿用上轮有限需求语言、提示词、工具、SQL AST 契约和 offline-hardened-v3；B 承接 summary/window/efg，C 承接 rank/game_count。选择不看 gold、Oracle 或分支执行结果。保留缺失/歧义/数据不足/失败状态，不进行 SQL fallback。比较基线是当前 `/agent/answer`、`routing_mode=model`，保留本地配置中的 Agent policy。

## 实现与验证

- [ ] 在 `tests/test_online_hybrid.py` 先写失败检查：单分支执行；工具失败不转 SQL；缺失输入不执行；阈值条件完整保留；请求超时终止 worker；评估门槛要求完整唯一记录，离线 replay 重算需求与数值。
- [ ] 新增 `scripts/two_stage_sql/online.py`：实验 ASGI 入口、请求内分流、阶段耗时和确定性结果展示；不依赖评估模块。
- [ ] 新增 `scripts/two_stage_sql/online_worker.py`：每臂独立持久进程，启动后收到的业务输入仅为 question；从同一仓库 `.env.local` 加载配置。父进程对每个请求实施 60 秒硬截止，超时终止进程并在下题重建。启动时间单列，不计入暖进程请求延迟。
- [ ] 新增 `scripts/evaluate_online_hybrid.py`：preflight / run / replay；交替 A/H 顺序、串行、每个输入独立解析；在两臂均结束后才调用 Oracle。记录实际调用、缓存/token、原始响应、分支、阶段和端到端时间、快照/来源 hash；唯一输出目录，不覆盖旧实验。
- [ ] 新增 `metadata/online_hybrid_v1.json`：24 个新问题（10 common / 8 expanded / 6 protection），每题两次。先用旧 development 12 题每题一次作硬化版本回归；不根据新题输出修改实现、提示词或预算。
- [ ] 跑离线测试和 preflight，保存冻结源码，然后先回归、再完整新留出运行。网络调用使用沙箱外执行；仅本地配置，不显示凭据。
- [ ] 运行 replay / 完整性与凭据审计，更新 `docs/research/2026-10-15-online-hybrid-experiment.md` 与 PTO 课程实验入口。旧分数和来源保持原样。

## 事前判定标准

这些是本次实验的工程研究门槛，不代表业务 SLA 或上线许可：

1. 完整 24×2×2=96 条 A/H 记录，无重复、缺失或伪造成功。
2. H common 20/20，且不低于 A；H protection 12/12；H 全部 accepted_wrong=0。
3. 8 个 expanded 场景至少 7 个两次均正确。
4. H common P95 ≤ A common P95×1.25；H 全集 P95 ≤ 5 秒。无正确性代价换速度。
5. 全部门槛通过也仅支持考虑结构化子范围的集成候选；完整 Agent 的多目标、跨赛季、RAG、多轮及状态能力未受此评估支持。

## 解释边界

“在线”指实验 ASGI 请求真实调用 provider、实时选路并执行，非离线选已存输出；不公开部署、不测 TCP/公网传输或并发。A/H 均由同一 ASGI 客户端发送，计时从父进程发送 question 到收到完整 HTTP 响应，含 IPC、解析、执行和序列化；bootstrap 与 Oracle 耗时另列。RAG/trace 在实验子进程初始化为不可用，避免索引锁，因此不能据此评价它们。

作者构造题目、重复两次有关联，不声称统计显著或开放世界覆盖。上轮已暴露的全体球员扫描预算问题不在本轮擅自优化。模型/SQLite 限制沿用 15 秒/2 秒，需求协议最多一次修复，SQL 不修复；进程截止是新增执行封装，不是放宽查询契约。
