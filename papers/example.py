# main.py
import os
import time
import agenta as ag
from agenta.sdk.types import PromptTemplate
from agenta.tracing.enums import Reference  # 可选：更直观地写 refs key
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool

# ========= 0) 初始化 Agenta =========
ag.init()  # 会读取 AGENTA_HOST / AGENTA_API_KEY

# ========= 1) Agenta Prompt 拉取工具 =========
def fetch_prompt(app_slug: str, environment_slug: str, inputs: dict) -> PromptTemplate:
    """
    从 Agenta 配置注册表获取 prompt 并格式化
    """
    config = ag.ConfigManager.get_from_registry(
        app_slug=app_slug,
        environment_slug=environment_slug
    )
    prompt = PromptTemplate(**config["prompt"]).format(**inputs)
    return prompt

# ========= 2) 用于记录“每一步”的子 span =========
@ag.instrument(spankind="chain")  # 在 Agenta UI 里会以“chain/step”分组展示
def log_task_step(step_index: int, task_name: str, output: str,
                  refs: dict | None = None, extra_meta: dict | None = None):
    """
    把某一步 Task 的信息写入 trace：
    - meta: 步骤索引、任务名等
    - internals: 截断的中间/长文本
    - metrics: 输出长度、耗时（如有）
    - event: 'task_completed'
    - refs: 关联到 Agenta 的 app/environment/variant 等资源
    """
    if refs:
        ag.tracing.store_refs(refs)  # 例如应用/环境 slug，便于 UI 过滤

    meta = {"step_index": step_index, "task_name": task_name}
    if extra_meta:
        meta.update(extra_meta)
    ag.tracing.store_meta(meta)

    # 避免把超长内容塞满 UI：仅存一段 preview
    preview = (output or "")[:1000]
    ag.tracing.store_internals({"output_preview": preview})

    ag.tracing.store_metrics({
        "output_len_chars": len(output or ""),
    })

    ag.tracing.get_current_span().add_event(
        name="task_completed",
        attributes={"task_name": task_name, "output_len": len(output or "")},
        namespace="events"
    )

# ========= 3) CrewAI + Tracing 的主流程 =========
@ag.instrument(spankind="workflow")  # 建立顶层 workflow span
def run_pipeline(topic: str, app_slug: str, environment_slug: str):
    # 在顶层 span 里写 refs，之后所有子 span 会自动共享上下文
    ag.tracing.store_refs({
        Reference.APPLICATION_SLUG.value: app_slug,
        Reference.ENVIRONMENT_SLUG.value: environment_slug,
    })

    # ------ 动态获取不同 Agent 的 Prompt ------
    research_prompt = fetch_prompt(app_slug, environment_slug, {"topic": topic})
    writer_prompt   = fetch_prompt(app_slug, environment_slug, {"topic": topic})

    # ------ 创建 Agents ------
    researcher = Agent(
        role="研究员",
        goal="基于指令进行资料调研与分析",
        backstory=(
            research_prompt.prompt["messages"][0]["content"]
            if research_prompt.prompt.get("messages") else ""
        ),
        tools=[ScrapeWebsiteTool()],
        allow_delegation=False,
        verbose=True
    )

    writer = Agent(
        role="写作者",
        goal=(
            writer_prompt.prompt["messages"][0]["content"]
            if writer_prompt.prompt.get("messages") else ""
        ),
        backstory="将调研内容转化为高质量文章",
        tools=[],
        allow_delegation=False,
        verbose=True
    )

    # ------ 定义任务链 ------
    task1 = Task(
        description="围绕主题进行调研，输出要点与参考资料",
        agent=researcher
    )
    task2 = Task(
        description="将调研结果整理成结构化文章",
        agent=writer
    )

    # ------ 用 callback 收集每步输出 ------
    class StepCollector:
        def __init__(self):
            self.steps = []
        def __call__(self, event):
            # CrewAI 的事件名可能因版本略有不同，示例用 task_completed
            if event.get("type") == "task_completed":
                self.steps.append({
                    "task": event.get("task_name") or "unknown_task",
                    "output": event.get("output") or ""
                })

    collector = StepCollector()

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True,
        callbacks=[collector]
    )

    # 记录执行前的 meta（输入参数等）
    ag.tracing.store_meta({"topic": topic})

    t0 = time.time()
    final_output = crew.kickoff(inputs={"topic": topic})
    exec_ms = (time.time() - t0) * 1000
    ag.tracing.store_metrics({"workflow_exec_ms": exec_ms})

    # 将每一个步骤写入 Agenta trace（作为子 span）
    refs = {
        Reference.APPLICATION_SLUG.value: app_slug,
        Reference.ENVIRONMENT_SLUG.value: environment_slug,
    }
    for idx, step in enumerate(collector.steps, 1):
        log_task_step(
            step_index=idx,
            task_name=step["task"],
            output=step["output"],
            refs=refs,
            extra_meta={"topic": topic}
        )

    # 也可把最终结果放入顶层 span 的 outputs 区域（用 internals 或 meta 均可）
    ag.tracing.store_internals({"final_output_preview": (final_output or "")[:1000]})
    ag.tracing.store_metrics({"final_output_len_chars": len(final_output or "")})

    return {
        "final_output": final_output,
        "steps": collector.steps,
        "workflow_exec_ms": exec_ms
    }

# ========= 4) 运行入口 =========
if __name__ == "__main__":
    TOPIC = "CrewAI 与 Agenta 集成实践"
    APP_SLUG = "my-app"          # 换成你在 Agenta 的 app_slug
    ENV_SLUG = "production"      # 或 staging/dev

    result = run_pipeline(TOPIC, APP_SLUG, ENV_SLUG)

    # 控制台也打印一份步骤与最终结果
    print("\n========== 全部步骤输出 ==========")
    for i, s in enumerate(result["steps"], 1):
        print(f"\n--- 步骤 {i}: {s['task']} ---")
        print(s["output"])

    print("\n========== 最终输出 ==========")
    print(result["final_output"])
    print(f"\n(执行耗时: {result['workflow_exec_ms']:.1f} ms)")
