from pathlib import Path
import json
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} MANIFEST", file=sys.stderr)
        return 2
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    assert manifest["format"] == "attention.training_curriculum.v1"
    assert manifest["tokenizer_version"] == "attention.byte_utf8.v1"
    stages = manifest["stages"]
    training = [stage for stage in stages if not stage["evaluation_only"]]
    evaluation = [stage for stage in stages if stage["evaluation_only"]]
    assert [stage["stage_id"] for stage in training] == ["stage0_debug", "stage1_general", "stage2_amharic", "stage3_oasst", "stage4_aya"]
    assert training[0]["parent_checkpoint"] is None
    for previous, current in zip(training, training[1:]):
        assert current["parent_checkpoint"] == previous["output_checkpoint"]
        assert current["output_checkpoint"]
    assert {stage["stage_id"] for stage in evaluation} == {"evaluation_pg19", "evaluation_aya_suite"}
    assert all(stage["parent_checkpoint"] is None and stage["output_checkpoint"] is None for stage in evaluation)
    gates = manifest["global_gates"]
    assert all(gates.values())
    print("training curriculum manifest validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
