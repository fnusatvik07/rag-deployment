"""Tests for app.embedding module."""

from unittest.mock import MagicMock, patch


class TestGetOrCreateIndex:
    @patch("app.embedding._pc")
    def test_creates_index_when_missing(self, mock_pc):
        mock_pc.has_index.return_value = False
        mock_status = MagicMock()
        mock_status.status.get.side_effect = [False, True]
        mock_pc.describe_index.return_value = mock_status
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index

        from app.embedding import _get_or_create_index

        result = _get_or_create_index()

        mock_pc.create_index_for_model.assert_called_once()
        assert result == mock_index

    @patch("app.embedding._pc")
    def test_returns_existing_index(self, mock_pc):
        mock_pc.has_index.return_value = True
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index

        from app.embedding import _get_or_create_index

        result = _get_or_create_index()

        mock_pc.create_index_for_model.assert_not_called()
        assert result == mock_index


class TestUpsertChunks:
    @patch("app.embedding._get_or_create_index")
    def test_upserts_single_batch(self, mock_get_index):
        mock_index = MagicMock()
        mock_get_index.return_value = mock_index

        from app.embedding import upsert_chunks

        records = [
            {"id": "a::chunk-0", "chunk_text": "text0", "source": "a.pdf", "pages": "1"},
            {"id": "a::chunk-1", "chunk_text": "text1", "source": "a.pdf", "pages": "2"},
        ]
        count = upsert_chunks(records)

        assert count == 2
        mock_index.upsert_records.assert_called_once()

    @patch("app.embedding._get_or_create_index")
    def test_upserts_multiple_batches(self, mock_get_index):
        mock_index = MagicMock()
        mock_get_index.return_value = mock_index

        from app.embedding import upsert_chunks

        records = [{"id": f"a::chunk-{i}", "chunk_text": f"text{i}", "source": "a.pdf"} for i in range(100)]
        count = upsert_chunks(records, batch_size=30)

        assert count == 100
        # 100 records / 30 batch = 4 batches (30+30+30+10)
        assert mock_index.upsert_records.call_count == 4

    @patch("app.embedding._get_or_create_index")
    def test_empty_records(self, mock_get_index):
        mock_index = MagicMock()
        mock_get_index.return_value = mock_index

        from app.embedding import upsert_chunks

        count = upsert_chunks([])
        assert count == 0
        mock_index.upsert_records.assert_not_called()

    @patch("app.embedding._get_or_create_index")
    def test_record_format(self, mock_get_index):
        mock_index = MagicMock()
        mock_get_index.return_value = mock_index

        from app.embedding import upsert_chunks

        records = [
            {"id": "doc::chunk-0", "chunk_text": "hello", "source": "doc.pdf", "pages": "3"},
        ]
        upsert_chunks(records)

        call_args = mock_index.upsert_records.call_args
        batch = call_args[0][1]  # second positional arg is the batch
        assert batch[0]["_id"] == "doc::chunk-0"
        assert batch[0]["chunk_text"] == "hello"
        assert batch[0]["source"] == "doc.pdf"
        assert batch[0]["pages"] == "3"
