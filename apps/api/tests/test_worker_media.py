from app.workers.jobs import _group_webm_chunks


def test_mediarecorder_chunks_are_reassembled_and_page_streams_stay_separate() -> None:
    header = b"\x1a\x45\xdf\xa3"

    streams = _group_webm_chunks([
        header + b"page-one-first",
        b"page-one-second",
        header + b"page-two-first",
        b"page-two-second",
    ])

    assert streams == [
        header + b"page-one-firstpage-one-second",
        header + b"page-two-firstpage-two-second",
    ]
