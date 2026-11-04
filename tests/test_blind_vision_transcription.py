import base64

from ilm.visual_lm.blind_vision_transcription import OPTIONS,PROMPT,parse_response,request_body


def test_request_accepts_only_pixels_and_fixed_instruction():
    data=b"literal image bytes"
    body=request_body(data);message=body["messages"][0]
    assert set(message)=={"role","content","images"}
    assert message["content"]==PROMPT and base64.b64decode(message["images"][0])==data
    assert body["options"]==OPTIONS and body["think"] is False and body["stream"] is False
    body["options"]["seed"]=-1
    assert request_body(data)["options"]["seed"]==20266371


def test_malformed_recognition_is_not_an_empty_correct_answer():
    good=parse_response({"message":{"content":'{"text":"汉字","readable":true}'}})
    assert good["valid_json"] and good["text"]=="汉字"
    for response in ({},{"message":None},{"message":{"content":"not JSON"}},
                     {"message":{"content":'{"text":"","readable":"false"}'}},
                     {"message":{"content":'{"text":"","readable":false,"extra":"answer"}'}}):
        invalid=parse_response(response)
        assert not invalid["valid_json"] and invalid["text"] is None
