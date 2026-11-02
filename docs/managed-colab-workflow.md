# Colab自动执行与持久保存

日期：2026-11-02。用户要求先修好执行与保存，确认能跑通后才继续训练。旧P0累计约1.82/2 CU，尚未通过；本流程不重置历史消耗。

## 修复内容

- 一个代码单元格完成全部阶段，取消“准备完成后再由浏览器启动第二单元格”的依赖。源码随入口打包并校验摘要，不依赖GPU连接后临时推送代码。
- 原始日志、逐阶段报告、状态、检查点和摘要直接写入`MyDrive/DiffusionLLM/runs/<唯一运行ID>/`。不依赖浏览器下载，不覆盖既有运行。数据和原始基座仍在临时计算目录。
- 检查子进程退出码、完成标志、训练步数、恢复一致性、文件大小及SHA-256；退出码为零但报告为partial、文件缺失或摘要不符也停止。
- 进程超时后终止其Linux进程组；后续阶段不启动。使用固定全局截止时间，写状态所花时间也占预算。
- 常规异常、预检失败及中断都请求释放运行时。Colab入口另启动独立于Drive I/O及笔记本内核的看门狗进程；正常结束前执行有界的Drive刷新，再请求释放。
- 释放请求返回与资源面板的0活动会话是两件事。Drive挂载内读回与会话外云端读回也分别记录。

官方接口依据：[runtime.unassign源码](https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py)、[drive.flush_and_unmount源码](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py)。保护是尽力而为；以实际资源面板和外部文件读回为准。

## 本地验证

```sh
python -s -m coverage run --source=experiments.managed_run -m pytest tests -q
python -s -m coverage report -m
python -s -m experiments.colab_pipeline --mode transport --durable-root private/workflow-validation --run-id UNIQUE_ID
python -s -m experiments.build_colab_notebook
```

已运行：全套39项测试通过；`managed_run.py`语句覆盖率93%；相关新增文件ruff检查通过。独立进程实际执行写文件、读回、错误退出及超时。真实微型随机网络的参数更新及精确恢复测试沿用原测试；不是0.5B权重训练结果。

审查发现并修复了：预检失败不释放、Drive写入会阻塞单线程计时、状态写入延长截止时间，以及全新源码目录缺少`results/`导致数据准备失败。新增空工作目录准备测试已经先复现失败再通过修复。

本地Windows未覆盖Linux的`killpg`/`/proc`分支。云端CPU测试会实际检查超时子孙进程终止；回执产生前不声称已验证。

## 云端验证及训练门槛

入口：[CPU流程验证笔记本](https://colab.research.google.com/drive/1Fx7uVAG8jQwMibXYfvDTK3k8ZonXjTab)。可重建副本为`notebooks/managed-p0.ipynb`。默认CPU，`MODE="transport"`；使用明确标记的JSON状态文件，不下载模型、不运行神经网络训练。

CPU验证包含自动顺序执行、下一独立进程读回、保存校验、报错停止、超时终止进程组、Drive刷新及会话结束。最后一关是从Colab运行时以外读取`transport-receipt.json`及实际文件，逐项核对摘要。

挂载Drive时Google可能要求用户亲自授权。挂载后笔记本代码可以访问Drive文件；本实现只使用上述项目目录。不在未授权时自动操作确认页。[Colab挂载说明](https://research.google.com/colaboratory/faq.html#drive-timeout)。

本文件形成时：云端CPU验证已请求启动，仍待完整回执及外部读回；GPU未重开。后续以实际回执补充状态。

只有以下条件同时满足，才配置P0模式：

1. CPU云端回执和实际文件已从外部读回并通过；保存回执精确SHA-256。
2. 回执对应源码包摘要与将运行代码完全一致；改代码后重做验证。
3. 核实当前设备、费率和余额；累计预算包含前两次约1.82 CU。现有P0剩余不足600秒最低窗口，默认会拒绝训练，不擅自增加原2 CU上限。
4. 结果目录确为挂载Drive，报告的可用空间不少于24 GiB；该报告不保证Drive账户配额，真实保存失败仍停止。

P0通过也不代表本次检查点可直接用`full_parameter_pilot --resume`进入P1：两入口身份清单不同。P1须核对兼容恢复接口，不能删除身份校验绕过。当前只修复并验证P0执行/保存，不改变训练目标、数据、学习率或成熟7B评测计划。
