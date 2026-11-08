# P2成对画布执行接续（2026-11-06）

**本地验证：**子集与fake集成回归为213 passed、1 deselected、0 failed、0 skipped（exit 0），ruff通过。唯一排除项为既有 `tests/test_checkpoint_restore_check.py::test_actual_restore_from_independent_cpu_process`；此前全量运行在该慢速子进程测试期间被手动中止，因此该项未验证，也不能称为超时或通过。独立Python与代码审查无剩余高置信问题，APPROVE仅限源码，不代表云端运行通过。

候选notebook已上传，Drive完整外部字节读回与本地base64解码逐字节相等；上传回执见 `private/runs/p2-execution-2026-11-06/notebook-upload-receipt.json`。上传时快照为 `RUN=False`；该阶段只上传notebook，没有启动GPU、训练或上传P2模型检查点。之后的L4主入口运行状态见下方实时接续记录。

后续实测状态（2026-11-06）：统一CUA浏览器控制已恢复，无需新聊天。CPU标准runtime通路输出`P2_CPU_CHANNEL_OK`并成功释放；GPU通路检查输出`P2_GPU_CHANNEL_OK`，曾实连NVIDIA A100-SXM4-80GB、速率8.88 CU/h，随后`runtime.unassign()`成功。两项仅验证通路，不是模型训练。15:44:12.266Z，P2主入口在NVIDIA L4启动，入口自身guard也启动；独立preflight guard于15:34:14Z启动，绝对截止16:33:14Z（epoch 1793982794.302804）。L4入口停在Google认证，用户确认没有可点击的授权窗口；Colab UI断开并删除运行时后实读180.16 CU、0活动会话、0 CU/h，阶段显示2.22/4 CU、余1.78 CU。之后Edge上的CPU-only认证探针通过：stdout含`WATCHDOG_STARTED_180_SECONDS`、`AUTH_START`、ADC凭证类型、`RUNS_FOLDER_NAME runs`、`AUTH_PROBE_OK`、`RELEASE_REQUESTED`；只读foldername，无凭据值输出，180秒guard启动，资源页复读确认释放。旧IAB重复探针标签已关闭。该验证只证明CPU认证和Drive foldername读取，不是P2训练；模型更新未观察到。CUA阶段summary已完成，停止截图为`private/runs/p2-execution-2026-11-06/stopped-after-auth.png`。远端原notebook修改时间`2026-11-06T16:06:39.302Z`、大小49435 bytes，与旧UI缓冲区自动保存冲突；原件未强制覆盖，快照base64待解码且修改者未知。现正准备新的唯一P2 Edge副本，wrapper仅认证超时和日志，使用历史venv且不pip install；研究bundle与9个受保护文件未改，暂未新启GPU或开始模型训练。完整事件见`private/runs/p2-execution-2026-11-06/live-budget.json`。

## 本地实现范围

- `actual_valid` 与 `masked_padding_canvas` 两种后缀政策分别从同一冻结基座开始，使用相同训练/开发集、更新顺序、随机种子和成对掩码；容量为128。短测分别计时并丢弃，随后在看新结果前锁定两组共同更新量，上限100步。
- 两组主比较使用相同的固定画布条件计算开发集CE；各自训练政策下的CE另行报告。完整报告还保留生成样本、EOS、空/短输出、重复及成本信息。
- 新实验身份绑定后缀政策、容量、数据、模型和源码摘要。每组通过Drive API保存完整检查点、核验服务器对象，并在独立CPU进程中下载恢复后做有限推理检查。以上是实现路径，尚待最终回归和真实云端验收。
- 检查点保存CUDA RNG状态；独立CPU恢复只校验该状态的结构，不重放CUDA RNG。因此不能据此声称GPU随机状态恢复或精确GPU续跑已验证。

## 候选notebook与构建命令

候选文件：`private/runs/p2-execution-2026-11-06/P2-paired-canvas.ipynb`。它嵌入的输入证明仅包含training和development，不包含300条保留续写或60条校准来源；Drive运行父目录已接入，代码为每次运行创建新的P2子目录。上传快照默认 `RUN=False`，且已完成外部全字节读回比对。之后L4主入口启动并停在Google认证，最终由Colab UI断开并删除运行时；没有模型参数更新证据。独立Edge CPU认证探针现已通过并释放，但不是模型运行；当前`RUN`状态未重新确认。远端原notebook元数据读取到修改时间`2026-11-06T16:06:39.302Z`、大小49435 bytes，与旧UI缓冲区自动保存状态冲突；原件未强制覆盖，快照base64待解码，修改者未知。新的唯一P2 Edge副本正在准备，研究bundle和9个受保护文件未改。

如需重建，请改用一个尚不存在的输出路径；builder以独占创建方式写文件，不覆盖已有notebook：

```powershell
python -s experiments/build_p2_paired_notebook.py "<new-output-path.ipynb>" --proof private/runs/p2/development-inputs-proof.json --parent-folder "<authorized-p2-runs-parent-id>"
```

该命令只生成notebook，不连接Colab、不启动训练。已上传候选默认保持 `RUN=False`；上传本身不代表资源门槛通过或训练已开始。

## 启动前资源门槛与4 CU止损线

从Colab资源页面重新读取并记录全部活动会话、当前余额和实际费率；只有确认其它活动会话为0、目标GPU与本次session id清楚后，才可形成新阶段起点。运行前120秒内的观测至少包含：

```json
{
  "observed_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "start_balance": 0.0,
  "current_balance": 0.0,
  "cu_per_hour": 0.0,
  "reserve_cu": 0.0,
  "overhead_seconds": 0.0,
  "gpu_name": "observed GPU name",
  "session_id": "observed session id",
  "other_active_sessions": 0
}
```

余额和费率必须来自当时页面；历史182.41 CU不能用作当前值。记录观测UTC时间、阶段开始余额、启动前当前余额、CU/小时、保存与恢复预留、准备/失败/评估/保存恢复等非更新开销、GPU名、当前session id，并以资源页核实 `other_active_sessions=0`。notebook和入口均会拒绝过期或不完整记录；notebook要求观测距启动不超过120秒。

独立4 CU上限包含准备、依赖/模型/数据获取、两政策短测、正式配对更新、评估、失败重试、两份API检查点保存、服务器核验、独立CPU恢复和释放开销。短测不是额外预算，也不能从4 CU之外单列；须在保留保存恢复余量后锁定共同步数，任一阶段预估超限就停止。总预算仍为180 CU，原AR200/DLM200和冻结数据不覆盖。

**当前执行状态与后续验收：**L4 GPU训练runtime已经断开并删除，资源页确认180.16 CU、0活动会话、0 CU/h；阶段显示消耗2.22/4 CU、剩余1.78 CU，非逐项结算。该训练入口认证未完成，模型更新未观察到。独立Edge CPU认证探针已成功通过并释放：180秒guard启动，认证及Drive foldername读取成功，无secret值输出，资源页再次确认0活动会话；这不等于P2训练已开始。root已保存`private/runs/p2-execution-2026-11-06/stopped-after-auth.png`并完成CUA阶段summary。远端原notebook元数据与旧UI缓冲区自动保存状态冲突；原件未强制覆盖，快照base64待解码且修改者未知。新的唯一P2 Edge副本正在准备，wrapper仅认证超时和日志，使用历史venv、不pip install，研究bundle和9个受保护文件未改；在副本与并发/预算检查完成前不启动GPU。后续仍需重新核实运行时与预算，然后按4 CU独立上限完成配对更新、评估、两份API保存、服务器摘要核验和独立CPU恢复。当前未声称模型训练成功或检查点已上传。事件台账见`private/runs/p2-execution-2026-11-06/live-budget.json`。
