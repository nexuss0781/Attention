from pathlib import Path
import json
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} MANIFEST", file=sys.stderr)
        return 2
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    assert manifest["format"] == "attention.english_competency_curriculum.v1"
    assert manifest["language"] == "en"
    assert manifest["tokenizer_version"] == "attention.byte_utf8.v1"
    policy = manifest["session_policy"]
    assert all(policy.values())
    assert policy["max_documents_per_session"] > 0
    assert policy["max_packed_tokens_per_session"] > 0

    roles = manifest["dataset_roles"]
    role_ids = {role["dataset_id"] for role in roles}
    required_roles = {
        "fineweb_english",
        "fineweb_edu_english",
        "dolma_english",
        "oasst1_english",
        "ultrachat_english",
        "helpsteer2_english",
        "ambigqa_english",
    }
    assert role_ids == required_roles
    for role in roles:
        assert role["url"].startswith("https://")
        assert role["language_policy"]
        assert role["quality_policy"]

    stages = manifest["stages"]
    assert [stage["stage_id"] for stage in stages] == [
        "stage0_debug",
        "stage1_english_foundation",
        "stage2_english_educational",
        "stage3_english_instruction",
        "stage4_english_clarification",
        "stage5_english_preference",
    ]
    assert stages[0]["parent_checkpoint"] is None
    for previous, current in zip(stages, stages[1:]):
        assert current["parent_checkpoint"] == previous["output_checkpoint"]
        assert current["output_checkpoint"]
        assert current["dataset_id"] in role_ids or current["dataset_id"] == "local_fixed_debug_stream"
        assert current["competency_ids"]
        assert current["chunk_policy"]

    competency_ids = {competency["competency_id"] for competency in manifest["competencies"]}
    assert competency_ids
    assert all(competency["mastery_rule"] for competency in manifest["competencies"])
    assert all(set(stage["competency_ids"]).issubset(competency_ids) for stage in stages)

    gates = manifest["global_gates"]
    assert all(gates.values())
    print("English competency curriculum validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
