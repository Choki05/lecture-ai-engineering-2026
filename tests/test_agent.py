"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    # save_learning_log は承認待ちで止まるので、承認が不要なツールだけ呼ばせる
    model = TestModel(call_tools=sorted(BASE_TOOLS))
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    assert len(_parts(result, ToolReturnPart)) == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_save_learning_log_waits_for_approval(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LOG_DIR", tmp_path)  # 本物の learning_log/ を汚さない
    model = TestModel(call_tools=["save_learning_log"])

    with agent.override(model=model):
        result = agent.run_sync("今日の学びをメモして")

    # 承認待ちで止まり、まだ何も保存されていない
    assert isinstance(result.output, DeferredToolRequests)
    assert [c.tool_name for c in result.output.approvals] == ["save_learning_log"]
    assert list(tmp_path.iterdir()) == []

    # 承認すると保存される
    with agent.override(model=model):
        agent.run_sync(
            None,
            message_history=result.all_messages(),
            deferred_tool_results=result.output.build_results(approve_all=True),
        )
    assert len(list(tmp_path.iterdir())) == 1


def test_save_learning_log_denied_does_not_write(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LOG_DIR", tmp_path)
    model = TestModel(call_tools=["save_learning_log"])

    with agent.override(model=model):
        first = agent.run_sync("メモして")
        agent.run_sync(
            None,
            message_history=first.all_messages(),
            deferred_tool_results=first.output.build_results(
                approvals={first.output.approvals[0].tool_call_id: False}
            ),
        )
    assert list(tmp_path.iterdir()) == []
