from scripts.audit_native_pixel_transcripts_v61 import normalize,scores,distance


def test_transcription_normalization_preserves_case_and_punctuation():
    assert normalize("e\u0301 中 文") == "é中文"
    assert scores("Word",["word"])["text_exact"] == 0
    assert scores("中文。",["中文"])["text_exact"] == 0


def test_extra_missing_and_alternate_text_are_scored_without_repair():
    assert scores("中",["中国"])["normalized_character_error"] == .5
    assert scores("",["中国"])["bounded_character_accuracy"] == 0
    assert scores("中国XYZ",["中国"])["normalized_character_error"] == 1.5
    assert scores("中华",["中国","中华"])["text_exact"] == 1
    assert distance("中国","国中") == 2
