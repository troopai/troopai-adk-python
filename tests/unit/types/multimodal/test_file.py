"""Tests for the multimodal ``File`` and ``Image`` argument types.

Three concerns:
1. Construction enforces the exactly-one-carrier invariant.
2. Pydantic integration emits the right JSON schema and round-trips
   strings through ``_from_string``.
3. Subclass narrowing (``Image.ALLOWED_MIME_PREFIXES``) is enforced.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from troopai.adk.types.multimodal import File, Image


class TestFileConstruction:
    def test_url_only(self) -> None:
        f = File(url="https://example.com/x.bin")
        assert f.url == "https://example.com/x.bin"
        assert f.data is None
        assert f.path is None

    def test_path_only(self) -> None:
        f = File(path=Path("/tmp/x.bin"))
        assert f.path == Path("/tmp/x.bin")
        assert f.data is None
        assert f.url is None

    def test_data_only(self) -> None:
        f = File(data=b"\x00\x01", mime_type="application/octet-stream")
        assert f.data == b"\x00\x01"
        assert f.path is None
        assert f.url is None

    def test_no_carrier_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one of"):
            File()

    def test_two_carriers_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one of"):
            File(url="https://example.com/x", path=Path("/tmp/x"))

    def test_three_carriers_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one of"):
            File(url="https://example.com/x", path=Path("/tmp/x"), data=b"\x00")

    def test_empty_mime_raises(self) -> None:
        with pytest.raises(ValueError, match="mime_type must be non-empty"):
            File(url="https://example.com/x", mime_type="")

    def test_invalid_url_scheme_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="must start with one of"):
            File(url="ftp://example.com/x")

    def test_carrier_list_in_error_message(self) -> None:
        # Pin the cleaned-up "data, path, url" wording rather than
        # the raw f-string artifacts the first implementation emitted.
        with pytest.raises(ValueError, match=r"got 3 \(data, path, url\)"):
            File(data=b"x", path=Path("/tmp/x"), url="https://example.com/x")
        with pytest.raises(ValueError, match=r"got 0 \(none\)"):
            File()


class TestCarrierAccessors:
    """``kind()`` / ``read_bytes()`` — typed accessors that hide the
    nullable-trio branching from callers."""

    def test_kind_data(self) -> None:
        assert File(data=b"x").kind() == "data"

    def test_kind_path(self) -> None:
        assert File(path=Path("/tmp/x")).kind() == "path"

    def test_kind_url(self) -> None:
        assert File(url="https://example.com/x").kind() == "url"

    def test_read_bytes_data(self) -> None:
        assert File(data=b"hello").read_bytes() == b"hello"

    def test_read_bytes_path(self, tmp_path: Path) -> None:
        target = tmp_path / "payload.bin"
        target.write_bytes(b"world")
        assert File(path=target).read_bytes() == b"world"

    def test_read_bytes_url_raises(self) -> None:
        with pytest.raises(ValueError, match="data- or path-backed"):
            File(url="https://example.com/x").read_bytes()


class TestImageMimeNarrowing:
    def test_image_accepts_image_mime(self) -> None:
        img = Image(url="https://example.com/x.png", mime_type="image/png")
        assert img.mime_type == "image/png"

    def test_image_accepts_no_mime(self) -> None:
        img = Image(url="https://example.com/x.png")
        assert img.mime_type is None

    def test_image_rejects_non_image_mime(self) -> None:
        with pytest.raises(ValueError, match="does not match any allowed prefix"):
            Image(url="https://example.com/x.bin", mime_type="application/octet-stream")


class TestPydanticIntegration:
    """Pydantic round-trips strings to File/Image and emits a URI string schema."""

    def test_json_schema_is_uri_string(self) -> None:
        schema = TypeAdapter(File).json_schema()
        assert schema["type"] == "string"
        assert schema["format"] == "uri"
        assert "File" in schema["description"]

    def test_image_json_schema_says_image(self) -> None:
        schema = TypeAdapter(Image).json_schema()
        assert "Image" in schema["description"]

    def test_validate_http_url_string(self) -> None:
        ta = TypeAdapter(File)
        f = ta.validate_python("https://example.com/x.bin")
        assert f.url == "https://example.com/x.bin"

    def test_validate_local_path_string(self) -> None:
        ta = TypeAdapter(File)
        f = ta.validate_python("/tmp/x.bin")
        assert f.path == Path("/tmp/x.bin")
        assert f.url is None

    def test_validate_data_url_string(self) -> None:
        payload = base64.b64encode(b"\x89PNG\r\n").decode("ascii")
        ta = TypeAdapter(Image)
        img = ta.validate_python(f"data:image/png;base64,{payload}")
        assert img.data == b"\x89PNG\r\n"
        assert img.mime_type == "image/png"

    def test_validate_data_url_with_extra_params_in_any_order(self) -> None:
        # RFC 2397 allows ``;charset=…`` and other params; the regex
        # must locate ``;base64`` regardless of order, not assume it
        # comes last.
        payload = base64.b64encode(b"hi").decode("ascii")
        ta = TypeAdapter(File)
        f1 = ta.validate_python(f"data:text/plain;charset=utf-8;base64,{payload}")
        assert f1.data == b"hi"
        f2 = ta.validate_python(f"data:text/plain;base64;charset=utf-8,{payload}")
        assert f2.data == b"hi"

    def test_validate_non_base64_data_url_keeps_url(self) -> None:
        # Plain (non-base64) data URLs are valid per RFC 2397 but
        # rare; we keep them URL-backed and document the fall-through.
        ta = TypeAdapter(File)
        f = ta.validate_python("data:text/plain,hello")
        assert f.url == "data:text/plain,hello"
        assert f.data is None

    def test_validate_passes_through_existing_instance(self) -> None:
        ta = TypeAdapter(File)
        original = File(url="https://example.com/x.bin")
        out = ta.validate_python(original)
        # Identity isn't guaranteed by core_schema, but value MUST match
        assert out.url == original.url

    def test_validate_empty_string_raises(self) -> None:
        ta = TypeAdapter(File)
        with pytest.raises(Exception, match="empty string"):
            ta.validate_python("")

    def test_validate_wrong_type_raises(self) -> None:
        ta = TypeAdapter(File)
        with pytest.raises(Exception):
            ta.validate_python(123)


class TestAsDataUrl:
    def test_data_backed_round_trip(self) -> None:
        f = File(data=b"hello", mime_type="text/plain")
        assert f.as_data_url() == f"data:text/plain;base64,{base64.b64encode(b'hello').decode()}"

    def test_url_backed_existing_data_url_passthrough(self) -> None:
        original = "data:image/png;base64,AAAA"
        f = File(url=original)
        assert f.as_data_url() == original

    def test_url_backed_http_raises(self) -> None:
        f = File(url="https://example.com/x.bin")
        with pytest.raises(ValueError, match="data- or path-backed"):
            f.as_data_url()

    def test_data_default_mime(self) -> None:
        f = File(data=b"x")
        assert f.as_data_url().startswith("data:application/octet-stream;base64,")

    def test_subclass_default_mime_satisfies_prefix(self) -> None:
        # A mime-less data-backed ``Image`` MUST emit a MIME that its
        # own ``ALLOWED_MIME_PREFIXES`` accepts, otherwise the data-URL
        # cannot be read back. ``File``'s generic
        # ``application/octet-stream`` does not start with ``image/``.
        img = Image(data=b"x")
        assert img.as_data_url().startswith("data:image/")


class TestSerializeRoundTrip:
    """A serialized instance MUST deserialize back into an equivalent
    instance for every carrier shape — including a mime-less, data- or
    path-backed subclass with a narrowed MIME contract (the case that
    previously crashed on reload)."""

    def test_mime_less_data_backed_image_round_trips(self) -> None:
        # Regression: ``as_data_url`` emitted ``application/octet-stream``
        # while ``Image`` requires an ``image/`` prefix, so the data-URL
        # it produced could not be validated back.
        ta = TypeAdapter(Image)
        original = Image(data=b"\x89PNG\r\n")
        dumped = ta.dump_python(original, mode="json")
        restored = ta.validate_python(dumped)
        assert isinstance(restored, Image)
        assert restored.data == b"\x89PNG\r\n"
        assert restored.mime_type is not None
        assert restored.mime_type.startswith("image/")

    def test_mime_less_path_backed_image_round_trips(self, tmp_path: Path) -> None:
        target = tmp_path / "payload.png"
        target.write_bytes(b"JPEGDATA")
        ta = TypeAdapter(Image)
        dumped = ta.dump_python(Image(path=target), mode="json")
        restored = ta.validate_python(dumped)
        assert isinstance(restored, Image)
        assert restored.path == target

    def test_mime_less_data_backed_file_round_trips(self) -> None:
        # The unrestricted parent keeps its generic binary default.
        ta = TypeAdapter(File)
        dumped = ta.dump_python(File(data=b"hi"), mode="json")
        restored = ta.validate_python(dumped)
        assert restored.data == b"hi"
        assert restored.mime_type == "application/octet-stream"


class TestFunctionToolIntegration:
    """The end-to-end goal: declaring ``image: Image`` in a function
    signature MUST produce a ``"type": "string", "format": "uri"``
    schema for the LLM and reconstruct ``Image`` from the LLM's
    string argument before the function body runs.
    """

    async def test_function_tool_schema_is_uri_string(self) -> None:
        from troopai.adk.tools import function_tool

        @function_tool(name="ocr", description="Extract text from an image.")
        def ocr(image: Image) -> str:
            return image.url or str(image.path) or ""

        schema = ocr.get_json_schema()
        image_param = schema["properties"]["image"]
        assert image_param["type"] == "string"
        assert image_param["format"] == "uri"

    async def test_function_tool_reconstructs_image_from_string(self) -> None:
        from troopai.adk.tools import function_tool
        from troopai.adk.tools.tool_context import ToolContext

        captured: list[Image] = []

        @function_tool(name="ocr", description="Extract text from an image.")
        def ocr(image: Image) -> str:
            captured.append(image)
            return "ok"

        raw = '{"image": "https://example.com/x.png"}'
        ctx = ToolContext(
            tool_name="ocr",
            tool_call_id="c1",
            tool_arguments={"image": "https://example.com/x.png"},
            raw_arguments=raw,
            context={},
        )
        assert ocr.on_invoke is not None
        result = await ocr.on_invoke(ctx, raw)
        assert result == "ok"
        assert len(captured) == 1
        assert isinstance(captured[0], Image)
        assert captured[0].url == "https://example.com/x.png"
