"""Final eval: dữ liệu phải thật sự held-out và target phải là token kế tiếp."""

from __future__ import annotations

import pytest

from luna_zero.eval import (
    bat_buoc_tap_eval_sach,
    cua_so_du_doan,
    kiem_tra_tach_biet,
)


def test_final_eval_khong_duoc_trung_bat_ky_doc_raw_nao() -> None:
    """Quét TOÀN raw corpus, không chỉ train split: val cũng đã dùng chọn best.pt."""
    eval_docs = ["Một bài hoàn toàn mới.", "Bài held-out thứ hai."]
    raw_da_dung = ["Tài liệu train.", "Một bài hoàn toàn mới.", "Tài liệu validation."]
    kiem = kiem_tra_tach_biet(eval_docs, raw_da_dung)
    assert kiem.n_trung_train == 1
    with pytest.raises(ValueError, match="CONTAMINATION"):
        bat_buoc_tap_eval_sach(kiem)


def test_final_eval_tach_biet_thi_duoc_chap_nhan() -> None:
    kiem = kiem_tra_tach_biet(
        ["Held-out A", "Held-out B"],
        ["Corpus cũ A", "Corpus cũ B"],
    )
    assert kiem.hop_le
    bat_buoc_tap_eval_sach(kiem)


def test_final_eval_khong_duoc_lap_mau_noi_bo() -> None:
    kiem = kiem_tra_tach_biet(["A", "B", "A"], ["C"])
    assert kiem.n_trung_noi_bo_eval == 1
    with pytest.raises(ValueError, match="trùng nội bộ"):
        bat_buoc_tap_eval_sach(kiem)


def test_hash_contamination_khong_lach_duoc_bang_nfd_nfc() -> None:
    import unicodedata

    nfd = unicodedata.normalize("NFD", "Tiếng Việt")
    kiem = kiem_tra_tach_biet([nfd], ["Tiếng Việt"])
    assert kiem.n_trung_train == 1


def test_cua_so_eval_du_doan_token_ke_tiep() -> None:
    ids = [10, 11, 12, 13, 14, 15, 16]
    windows = list(cua_so_du_doan(ids, block_size=3))
    assert windows == [
        ([10, 11, 12], [11, 12, 13]),
        ([13, 14, 15], [14, 15, 16]),
    ]
    for x, y in windows:
        assert y[:-1] == x[1:]


def test_tao_heldout_tu_choi_source_nam_trong_raw(tmp_path) -> None:
    import json

    from luna_zero.eval import tao_corpus_heldout

    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "source.jsonl"
    source.write_text(json.dumps({"text": "Mẫu mới"}, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bên trong raw corpus"):
        tao_corpus_heldout(source, tmp_path / "final.jsonl", raw)


def test_tao_heldout_ghi_atomic_va_metadata_co_the_truy_nguyen(tmp_path) -> None:
    import json

    from luna_zero.data import iter_jsonl_texts
    from luna_zero.eval import file_fingerprint, tao_corpus_heldout

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "train.jsonl").write_text(
        json.dumps({"text": "Corpus cũ"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    source = tmp_path / "ngoai_repo.jsonl"
    source.write_text(
        "".join(
            json.dumps({"text": text}, ensure_ascii=False) + "\n"
            for text in ["Held-out một", "Held-out hai"]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "eval" / "final.jsonl"

    ket_qua = tao_corpus_heldout(source, output, raw)

    assert list(iter_jsonl_texts(output)) == ["Held-out một", "Held-out hai"]
    assert ket_qua.n_docs == 2
    assert ket_qua.kiem_tra_tach_biet.n_trung_train == 0
    assert ket_qua.source_fingerprint == file_fingerprint(source)
    assert ket_qua.output_fingerprint == file_fingerprint(output)
    assert not output.with_name(output.name + ".tmp").exists()


def test_tao_heldout_json_hong_phai_do_thay_vi_bo_qua(tmp_path) -> None:
    from luna_zero.eval import tao_corpus_heldout

    raw = tmp_path / "raw"
    raw.mkdir()
    source = tmp_path / "source.jsonl"
    source.write_text('{"text": "được"}\n{hong json}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="JSON hỏng"):
        tao_corpus_heldout(source, tmp_path / "final.jsonl", raw)


def test_human_final_bat_buoc_xac_nhan_nguon_do_nguoi_viet(tmp_path) -> None:
    import json

    from luna_zero.eval import ProvenanceHuman, tao_corpus_human_heldout

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "train.jsonl").write_text(
        json.dumps({"text": "Corpus cũ"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    source = tmp_path / "human.jsonl"
    source.write_text(
        json.dumps({"text": "Văn bản mới do người viết."}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    provenance = ProvenanceHuman(
        source_name="Nguồn thử",
        source_url="https://example.test/source",
        retrieved_at="2026-09-06",
        license_or_permission="Được phép dùng để eval",
    )

    with pytest.raises(ValueError, match="human-authored"):
        tao_corpus_human_heldout(
            source,
            tmp_path / "final_human.jsonl",
            raw,
            provenance,
            human_authored_confirmed=False,
        )


def test_human_final_meta_bat_thay_doi_corpus_sau_khi_khoa(tmp_path) -> None:
    import json

    from luna_zero.eval import (
        ProvenanceHuman,
        doc_meta_human_final,
        tao_corpus_human_heldout,
    )

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "train.jsonl").write_text(
        json.dumps({"text": "Corpus cũ"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    source = tmp_path / "human.jsonl"
    source.write_text(
        json.dumps({"text": "Một đoạn hoàn toàn mới."}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "final_human.jsonl"
    meta = tao_corpus_human_heldout(
        source,
        output,
        raw,
        ProvenanceHuman(
            source_name="Nguồn thử",
            source_url="https://example.test/source",
            retrieved_at="2026-09-06",
            license_or_permission="Được phép dùng để eval",
        ),
        human_authored_confirmed=True,
    )
    meta_path = tmp_path / "final_human.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    doc_meta_human_final(meta_path, output)

    output.write_text(
        json.dumps({"text": "Đã bị đổi sau khi khóa."}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="đã đổi"):
        doc_meta_human_final(meta_path, output)
