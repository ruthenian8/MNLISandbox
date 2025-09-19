from nli_groundedness.groundedness import aggregate_sentence_groundedness
from nli_groundedness.vlm_scorer import CaptionerAndLM


def test_stub_captioner_alignment():
    scorer = CaptionerAndLM(cap_ckpt="stub")
    text = "A man riding a bike along the street"
    ids_text, logprobs_text = scorer.token_logprobs_text_only(text)
    ids_cap, logprobs_cap = scorer.token_logprobs_captioner(None, text)

    assert ids_text.tolist() == ids_cap.tolist()
    assert len(logprobs_text) == len(logprobs_cap)

    tokenizer = scorer.processor.tokenizer
    result = aggregate_sentence_groundedness(
        logprobs_cap,
        logprobs_text,
        ids_text,
        tokenizer,
    )

    assert result.G_sentence_mean > 0
    assert len(result.token_diagnostics) == len(ids_text.tolist()[0]) - 1
    assert all(diag.token for diag in result.token_diagnostics)
